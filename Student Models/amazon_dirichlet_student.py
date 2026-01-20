#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.special import digamma
from collections import defaultdict
import evaluation
from llm_classifier_student import LLMClassifier
from llm_model_dirichlet_student import LLM



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

def store_dirichlet_uncertainty(dataset, epoch, probs):
    total,aleotoric,epistemic = compute_uncertainties(probs)
    UNCERTAINTY_BUFFER[dataset][epoch] = {
        "total_uncertainty": total.cpu().numpy(),
        "aleatoric_uncertainty": aleotoric.cpu().numpy(),
        "epistemic_uncertainty": epistemic.cpu().numpy(),
    }
    
def amazon_uncertainties(alpha, epoch):
    store_dirichlet_uncertainty("amazon", epoch, alpha)

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
    store_dirichlet_uncertainty("sst2", epoch, alpha)

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
    store_dirichlet_uncertainty("yahoo", epoch, alpha)

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
    store_dirichlet_uncertainty("youtube", epoch, alpha)

def evaluate():
    def dirichlet_to_prob(alpha):
        return alpha / alpha.sum(dim=1, keepdim=True)

    class TestDirichletDataset(Dataset):
        def __init__(self, samples, n_samples):
            self.samples = samples
            self.n_samples = n_samples

        def __len__(self):
            return self.n_samples

        def __getitem__(self, idx):
            return self.samples[idx]

    llm.model.eval()
    test_dataset = TestDirichletDataset(samples_test, len(samples_test))
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    def get_test_alpha(test_dataloader, classifier):
        all_alpha = []
        with torch.no_grad():
            for batch_samples in test_dataloader:
                alpha_batch = classifier.soft_labels_batch(input_texts=batch_samples)
                alpha_batch = torch.clamp(alpha_batch, min=1e-3)
                all_alpha.append(alpha_batch)
        return torch.cat(all_alpha, dim=0)

    alpha_test = get_test_alpha(test_dataloader, classifier)
    stu_probs = dirichlet_to_prob(alpha_test).cpu().numpy()

    f1_score = evaluation.compute_metric(gt_labels_test, stu_probs, metric='f1')
    ece = evaluation.compute_metric(gt_labels_test, stu_probs, metric='ece')
    acc = evaluation.compute_metric(gt_labels_test, stu_probs, metric='acc')
    nll = evaluation.compute_metric(gt_labels_test, stu_probs, metric='nll')
    brier = evaluation.compute_metric(gt_labels_test, stu_probs, metric='brier')

    print(
        'Student test f1-score: {}, Student test ECE: {}, '
        'Student test Accuracy: {}, Student test NLL: {}, '
        'Student test brier score: {}'.format(
            f1_score, ece, acc, nll, brier
        )
    )

    return alpha_test



def evaluate_train(epoch_alpha):
    probs_np = (epoch_alpha / epoch_alpha.sum(dim=1, keepdim=True)).cpu().numpy()

    print(
        "Student train f1-score:", evaluation.compute_metric(gt_labels_train, probs_np, "f1"),
        "Student train ECE:", evaluation.compute_metric(gt_labels_train, probs_np, "ece"),
        "Student train Accuracy:", evaluation.compute_metric(gt_labels_train, probs_np, "acc"),
        "Student train NLL:", evaluation.compute_metric(gt_labels_train, probs_np, "nll"),
        "Student train brier score:", evaluation.compute_metric(gt_labels_train, probs_np, "brier"),
    )



def train_student():

    dataset = DirichletDataset(samples_train)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    llm_params = list(filter(lambda p: p.requires_grad, llm.model.parameters()))

    if args.mode == "learnable":
        optimizer = optim.AdamW(
            [
                {"params": llm_params, "lr": args.lr},
                {"params": [a], "lr": args.lr_alpha0},
            ]
        )
    else:
        optimizer = optim.AdamW(llm_params, lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_dirichlet_loss = 0.0
        total_regularizer_loss = 0.0
        llm.model.train()
        epoch_alphas = []

        for batch_samples, batch_indices in dataloader:
            batch_probs = probs[batch_indices].to(llm.device)

            optimizer.zero_grad()

            alpha = classifier.soft_labels_batch(input_texts=batch_samples)
            alpha = torch.clamp(alpha, min=1e-3)

            if args.mode == "fixed":
                alpha0 = alpha.sum(dim=1, keepdim=True)
                alpha = alpha * (args.fixed_alpha0 / alpha0)

            dirichlet_loss_term = dirichlet_loss(alpha, batch_probs, weights)
            loss = dirichlet_loss_term

            if args.mode == "learnable":
                regularizer_loss = alpha0_l2_regularizer(alpha, a, args.beta)
                loss = loss + regularizer_loss
      

            loss.backward()
            optimizer.step()

            epoch_alphas.append(alpha.detach().cpu())
            total_loss += loss.item()
            if args.mode == "learnable":
                total_dirichlet_loss += dirichlet_loss_term.item()
                total_regularizer_loss += regularizer_loss.item()

        epoch_alpha = torch.cat(epoch_alphas, dim=0)

        if args.mode == "learnable":
            print(
                f"Epoch {epoch+1}/{args.epochs}, "
                f"Dirichlet Loss: {total_dirichlet_loss}, "
                f"Regularizer Loss: {total_regularizer_loss}, "
                f"Loss: {total_loss}, "
                f"alpha0_prior: {torch.exp(a).item()}"
            )
        else:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss}")

        evaluate_train(epoch_alpha)
        test_alpha = evaluate()
        amazon_uncertainties(test_alpha, epoch)
        sst2_uncertainties(epoch)
        yahoo_uncertainties(epoch)
        youtube_uncertainties(epoch)

    final_train_alphas = []
    llm.model.eval()
    with torch.no_grad():
        for batch_samples, _ in dataloader:
            alpha = classifier.soft_labels_batch(input_texts=batch_samples)
            alpha = torch.clamp(alpha, min=1e-3)

            if args.mode == "fixed":
                alpha0 = alpha.sum(dim=1, keepdim=True)
                alpha = alpha * (args.fixed_alpha0 / alpha0)

            final_train_alphas.append(alpha)

    final_train_alphas = torch.cat(final_train_alphas, dim=0)
    evaluate_train(final_train_alphas)

def save_uncertainty_buffer():
    torch.save(dict(UNCERTAINTY_BUFFER),"amazon_dirichlet_uncertainties.pt")

test_alpha = evaluate()
amazon_uncertainties(test_alpha, "pretrained")
sst2_uncertainties("pretrained")
yahoo_uncertainties("pretrained")
youtube_uncertainties("pretrained")
train_student()
save_uncertainty_buffer()



