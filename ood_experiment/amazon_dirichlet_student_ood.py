#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import logging
import builtins
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.special import digamma
from collections import defaultdict

from llm_classifier_modified import LLMClassifier
from llm_model_modified import LLM



parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["standard", "fixed", "learnable"], default="standard")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--fixed_alpha0", type=float, default=10.0)
parser.add_argument("--lr_alpha0", type=float, default=1e-3)
parser.add_argument("--beta", type=float, default=1.0)
args = parser.parse_args()

UNCERTAINTY_BUFFER = defaultdict(lambda: defaultdict(dict))

df_train = pd.read_csv("train_amazon.csv", header=None,nrows=15000)
df_test = pd.read_csv("test_amazon.csv", header=None,nrows=15000)

df_train = df_train.iloc[:10000]
df_test = df_test.iloc[:5000]

samples_train = df_train.iloc[:, 2].values
gt_labels_train = df_train.iloc[:, 0].values.astype(int)
samples_test = df_test.iloc[:, 2].values
gt_labels_test = df_test.iloc[:, 0].values.astype(int)

class PromptFormatting(object):
    def __init__(self):
        self.INSTRUCTION = "classify the sentiment of the Amazon review below into one of the following classes:"
        self.CLASSES = ["negative", "positive"]
        self.CLASSES_FOR_MATCHING = [self.CLASSES, ["neg", "pos"], ["1", "2"]]
        self.CLASSES_TEXT = "1. {}\n2. {}".format(self.CLASSES[0], self.CLASSES[1])

    def format_instruction(self, instruction):
        return "{}\n{}\n".format(instruction, self.CLASSES_TEXT)

    def format_content(self, content):
        return "review: {}\nthe review is ".format(content)

llm = LLM(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    use_reduced_precision=True,
    use_lora=True,
)

classifier = LLMClassifier(model=llm, prompt_formatting=PromptFormatting())

teacher_probs = torch.load("amazon_llora_teacher_probs.pt", map_location="cpu")
weights = torch.full((10000,), 1.0 / 10000.0).to(llm.device)

class DirichletDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx], idx

def dirichlet_loss(alpha, probs, weights):
    alpha0 = alpha.sum(dim=1, keepdim=True)
    log_gamma_alpha0 = torch.lgamma(alpha0)
    log_gamma_alpha = torch.lgamma(alpha).sum(dim=1, keepdim=True)
    weighted_log_probs = (alpha.unsqueeze(-1) - 1) * torch.log(probs + 1e-8)
    class_sum = weighted_log_probs.sum(dim=1)
    if weights.ndim == 1:
        weights = weights.unsqueeze(1)
    prompt_sum = (class_sum * weights.T).sum(dim=1, keepdim=True)
    return -(log_gamma_alpha0 - log_gamma_alpha + prompt_sum).mean()

a = torch.nn.Parameter(torch.tensor(2.3025851, device=llm.device))

def alpha0_l2_regularizer(alpha, a, beta):
    alpha0 = alpha.sum(dim=1)
    return beta * ((alpha0 - torch.exp(a)) ** 2).mean()

def compute_uncertainties(alpha):
    alpha0 = alpha.sum(dim=1, keepdim=True)
    probs = alpha / alpha0
    total = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    psi_alpha0 = digamma(alpha0 + 1.0)
    psi_alpha = digamma(alpha + 1.0)
    aleatoric = torch.sum(probs * (psi_alpha0 - psi_alpha), dim=1)
    epistemic = total - aleatoric
    return total, aleatoric, epistemic

def amazon_uncertainties(alpha, epoch):
    t, a_, e = compute_uncertainties(alpha)
    UNCERTAINTY_BUFFER["amazon"][epoch] = {
        "total_uncertainty": t.cpu().numpy(),
        "aleatoric_uncertainty": a_.cpu().numpy(),
        "epistemic_uncertainty": e.cpu().numpy(),
    }

class TestDirichletDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples
            

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

def sst2_uncertainties(epoch):
    df = pd.read_csv("test_sst2.csv", engine="python")
    samples = df.iloc[:, 1].astype(str).values
    class PF(object):
        def __init__(self):
            self.INSTRUCTION = "Select the sentiment category that best matches the opinion expressed in the review snippet."
            self.CLASSES = ["negative", "positive"]
            self.CLASSES_FOR_MATCHING = [self.CLASSES, ["neg", "pos"], ["1", "2"]]
            self.CLASSES_TEXT = "1. {}\n2. {}".format(*self.CLASSES)
        def format_instruction(self, i):
            return "{}\n{}\n".format(i, self.CLASSES_TEXT)
        def format_content(self, c):
            return "review: {}\nthe review is ".format(c)
    clf = LLMClassifier(model=llm, prompt_formatting=PF())
    dataset = TestDirichletDataset(samples)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    llm.model.eval()
    alphas = []
    with torch.no_grad():
        for b in loader:
            alphas.append(clf.soft_labels_batch(input_texts=b))
    alpha = torch.cat(alphas)
    t, a_, e = compute_uncertainties(alpha)
    UNCERTAINTY_BUFFER["sst2"][epoch] = {
        "total_uncertainty": t.cpu().numpy(),
        "aleatoric_uncertainty": a_.cpu().numpy(),
        "epistemic_uncertainty": e.cpu().numpy(),
    }

def yahoo_uncertainties(epoch):
    df = pd.read_csv("test_yahoo.csv", header=None)
    df = df.iloc[:5000]
    samples = (
        "Question: " + df.iloc[:, 1].astype(str) + " " + df.iloc[:, 2].astype(str)
        + "\nAnswer: " + df.iloc[:, 3].astype(str)
    ).values
    class PF(object):
        def __init__(self):
            self.INSTRUCTION = "Identify the topic that the following question and answer belong to:"
            self.CLASSES = [
                "Society & Culture","Science & Mathematics","Health","Education & Reference",
                "Computers & Internet","Sports","Business & Finance","Entertainment & Music",
                "Family & Relationships","Politics & Government",
            ]
            self.CLASSES_FOR_MATCHING = [self.CLASSES]
            self.CLASSES_TEXT = "\n".join([f"{i+1}. {c}" for i, c in enumerate(self.CLASSES)])
        def format_instruction(self, i):
            return f"{i}\n{self.CLASSES_TEXT}\n"
        def format_content(self, c):
            return f"{c}\nthe topic is "
    clf = LLMClassifier(model=llm, prompt_formatting=PF())
    dataset = TestDirichletDataset(samples)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    llm.model.eval()
    alphas = []
    with torch.no_grad():
        for b in loader:
            alphas.append(clf.soft_labels_batch(input_texts=b))
    alpha = torch.cat(alphas)
    t, a_, e = compute_uncertainties(alpha)
    UNCERTAINTY_BUFFER["yahoo"][epoch] = {
        "total_uncertainty": t.cpu().numpy(),
        "aleatoric_uncertainty": a_.cpu().numpy(),
        "epistemic_uncertainty": e.cpu().numpy(),
    }

def youtube_uncertainties(epoch):
    df = pd.read_csv("youtube.csv", engine="python")[1245:]
    samples = df.iloc[:, 3].astype(str).values
    class PF(object):
        def __init__(self):
            self.INSTRUCTION = "Judge whether the Youtube comment should be flagged as spam."
            self.CLASSES = ["not spam", "spam"]
            self.CLASSES_FOR_MATCHING = [self.CLASSES, ["ham", "spam"], ["0", "1"]]
            self.CLASSES_TEXT = "1. {}\n2. {}".format(*self.CLASSES)
        def format_instruction(self, i):
            return "{}\n{}\n".format(i, self.CLASSES_TEXT)
        def format_content(self, c):
            return "comment: {}\nthe comment is ".format(c)
    clf = LLMClassifier(model=llm, prompt_formatting=PF())
    dataset = TestDirichletDataset(samples)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    llm.model.eval()
    alphas = []
    with torch.no_grad():
        for b in loader:
            alphas.append(clf.soft_labels_batch(input_texts=b))
    alpha = torch.cat(alphas)
    t, a_, e = compute_uncertainties(alpha)
    UNCERTAINTY_BUFFER["youtube"][epoch] = {
        "total_uncertainty": t.cpu().numpy(),
        "aleatoric_uncertainty": a_.cpu().numpy(),
        "epistemic_uncertainty": e.cpu().numpy(),
    }

def evaluate():
    def to_prob(a): return a / a.sum(dim=1, keepdim=True)
    dataset = TestDirichletDataset(samples_test)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    llm.model.eval()
    alphas = []
    with torch.no_grad():
        for b in loader:
            alphas.append(classifier.soft_labels_batch(input_texts=b))
    alpha = torch.cat(alphas)
 
    return alpha

def train_student():
    dataset = DirichletDataset(samples_train)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    params = list(filter(lambda p: p.requires_grad, llm.model.parameters()))
    if args.mode == "learnable":
        opt = optim.AdamW([{"params": params, "lr": args.lr}, {"params": [a], "lr": args.lr_alpha0}])
    else:
        opt = optim.AdamW(params, lr=args.lr)
    for epoch in range(args.epochs):
        llm.model.train()
        for texts, idx in loader:
            opt.zero_grad()
            alpha = classifier.soft_labels_batch(input_texts=texts)
            alpha = torch.clamp(alpha, min=1e-3)
            if args.mode == "fixed":
                alpha = alpha * (args.fixed_alpha0 / alpha.sum(dim=1, keepdim=True))
            loss = dirichlet_loss(alpha, teacher_probs[idx].to(llm.device), weights)
            if args.mode == "learnable":
                loss = loss + alpha0_l2_regularizer(alpha, a, args.beta)
            loss.backward()
            opt.step()
        test_alpha = evaluate()
        amazon_uncertainties(test_alpha, epoch)
        sst2_uncertainties(epoch)
        yahoo_uncertainties(epoch)
        youtube_uncertainties(epoch)
        print(f"Epoch {epoch} completed: Amazon, SST2, Yahoo, YouTube uncertainties stored")

def export_excel():
    epochs = ["pretrained"] + list(range(args.epochs))
    with pd.ExcelWriter(
        "amazon_dirichlet_learnable_alpha0_llora_uncertainties_seed-2.xlsx",
        engine="xlsxwriter"
    ) as writer:
        for ds, ep_data in UNCERTAINTY_BUFFER.items():
            first = next(iter(ep_data.values()))
            n = len(first["total_uncertainty"])
            cols = {}
            for e in epochs:
                if e not in ep_data:
                    continue
                for k in ["total_uncertainty", "aleatoric_uncertainty", "epistemic_uncertainty"]:
                    cols[(str(e), k)] = ep_data[e][k]
            df = pd.DataFrame(cols)
            df.insert(0, "example_id", range(n))
            df.columns = pd.MultiIndex.from_tuples([("example_id", "")] + list(cols.keys()))
            df.to_excel(writer, sheet_name=ds, index=False)


def save_uncertainty_buffer():
    torch.save(
        dict(UNCERTAINTY_BUFFER),
        "amazon_dirichlet_learnable_alpha0_llora_uncertainties_seed-2.pt"
    )

test_alpha = evaluate()
amazon_uncertainties(test_alpha, "pretrained")
sst2_uncertainties("pretrained")
yahoo_uncertainties("pretrained")
youtube_uncertainties("pretrained")
print(f"Epoch pretrained completed: Amazon, SST2, Yahoo, YouTube uncertainties stored")
train_student()
save_uncertainty_buffer()
export_excel()
