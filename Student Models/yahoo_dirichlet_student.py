#!/usr/bin/env python
# coding: utf-8

import os
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.special import digamma

from llm_classifier_modified import LLMClassifier
from llm_model_modified_Copy2 import LLM
import evaluation

parser = argparse.ArgumentParser()
parser.add_argument("--mode",choices=["standard", "fixed", "learnable"],default="standard",help="Training mode: standard | fixed alpha0 | learnable alpha0")
parser.add_argument("--epochs",type=int,default=50,help="Number of training epochs")
parser.add_argument("--batch_size",type=int,default=1,help="Batch size for training")
parser.add_argument("--lr",type=float,default=1e-5,help="Learning rate for LLM parameters")
parser.add_argument("--fixed_alpha0",type=float,default=10.0,help="Fixed alpha0 value (used when mode=fixed)")
parser.add_argument("--lr_alpha0",type=float,default=1e-3,help="Learning rate for learnable alpha0 parameter a")
parser.add_argument("--beta",type=float,default=1.0,help="Regularization strength for learnable alpha0")

args = parser.parse_args()

df_train = pd.read_csv("train_yahoo.csv", header=None)
df_test = pd.read_csv("test_yahoo.csv", header=None)

n_train = 10000
n_test = 5000

df_train_actual = df_train.iloc[:n_train]
df_test_actual = df_test.iloc[:n_test]

def format_prompt(q1, q2, a):
    return "Question: " + q1.astype(str) + " " + q2.astype(str) + "\nAnswer: " + a.astype(str)

samples_train = format_prompt(
    df_train_actual.iloc[:, 1],
    df_train_actual.iloc[:, 2],
    df_train_actual.iloc[:, 3]
).values

samples_test = format_prompt(
    df_test_actual.iloc[:, 1],
    df_test_actual.iloc[:, 2],
    df_test_actual.iloc[:, 3]
).values

gt_labels_train = df_train_actual.iloc[:, 0].values.astype(int)
gt_labels_test = df_test_actual.iloc[:, 0].values.astype(int)


class PromptFormatting(object):
    def __init__(self):
        self.INSTRUCTION = "Identify the topic that the following question and answer belong to:"
        self.CLASSES = [
            "Society & Culture",
            "Science & Mathematics",
            "Health",
            "Education & Reference",
            "Computers & Internet",
            "Sports",
            "Business & Finance",
            "Entertainment & Music",
            "Family & Relationships",
            "Politics & Government"
        ]
        self.CLASSES_FOR_MATCHING = [self.CLASSES]
        self.CLASSES_TEXT = "\n".join([f"{i+1}. {c}" for i, c in enumerate(self.CLASSES)])

    def format_instruction(self, instruction):
        return f"{instruction}\n{self.CLASSES_TEXT}\n"

    def format_content(self, content):
        return f"{content}\nthe topic is "

llm = LLM(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    use_reduced_precision=True,
    use_lora=True
)

classifier = LLMClassifier(model=llm, prompt_formatting=PromptFormatting())


probs = torch.load("yahoo_llora_teacher_probs.pt", weights_only=False)
weights = torch.full((10000,), 1.0 / 10000.0, dtype=torch.float32).to(llm.device)


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
    alpha0_prior = torch.exp(a)
    return beta * ((alpha0 - alpha0_prior) ** 2).mean()


def compute_uncertainties(alpha):
    alpha0 = alpha.sum(dim=1, keepdim=True)
    probs = alpha / alpha0

    total_uncertainty = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    data_uncertainty = torch.sum(
        probs * (digamma(alpha0 + 1.0) - digamma(alpha + 1.0)), dim=1
    )
    knowledge_uncertainty = total_uncertainty - data_uncertainty

    return total_uncertainty, data_uncertainty, knowledge_uncertainty


def yahoo_uncertainties(alpha, epoch):
    tu, du, ku = compute_uncertainties(alpha)
    torch.save(
        {"total_uncertainty": tu, "data_uncertainty": du, "knowledge_uncertainty": ku},
        f"yahoo_uncertainties_seed{seed}_epoch{epoch}.pt"
    )

def amazon_uncertainties(epoch):
    df = pd.read_csv("test_amazon.csv", header=None)
    samples = df.iloc[:5000, 2].values

    class PF(object):
        def __init__(self):
            self.INSTRUCTION = "classify the sentiment of the Amazon review below into one of the following classes:"
            self.CLASSES = ["negative", "positive"]
            self.CLASSES_TEXT = "1. negative\n2. positive"

        def format_instruction(self, instruction):
            return f"{instruction}\n{self.CLASSES_TEXT}\n"

        def format_content(self, content):
            return f"review: {content}\nthe review is "

    clf = LLMClassifier(model=llm, prompt_formatting=PF())
    loader = DataLoader(samples, batch_size=16, shuffle=False)

    alphas = []
    with torch.no_grad():
        for b in loader:
            alphas.append(clf.soft_labels_batch(b))

    alpha = torch.cat(alphas, dim=0)
    tu, du, ku = compute_uncertainties(alpha)
    torch.save({"total": tu, "data": du, "knowledge": ku},
               f"amazon_uncertainties_epoch{epoch}.pt")

def sst2_uncertainties(epoch):
    df = pd.read_csv("test_sst2.csv")
    samples = df.iloc[:, 1].values

    class PF(object):
        def __init__(self):
            self.INSTRUCTION = "Select the sentiment category that best matches the opinion expressed in the review snippet."
            self.CLASSES_TEXT = "1. negative\n2. positive"

        def format_instruction(self, instruction):
            return f"{instruction}\n{self.CLASSES_TEXT}\n"

        def format_content(self, content):
            return f"review: {content}\nthe review is "

    clf = LLMClassifier(model=llm, prompt_formatting=PF())
    loader = DataLoader(samples, batch_size=16, shuffle=False)

    alphas = []
    with torch.no_grad():
        for b in loader:
            alphas.append(clf.soft_labels_batch(b))

    alpha = torch.cat(alphas, dim=0)
    tu, du, ku = compute_uncertainties(alpha)
    torch.save({"total": tu, "data": du, "knowledge": ku},
               f"sst2_uncertainties_epoch{epoch}.pt")

def youtube_uncertainties(epoch):
    df = pd.read_csv("youtube.csv")
    samples = df.iloc[:, 3].values

    class PF(object):
        def __init__(self):
            self.INSTRUCTION = "Judge whether the Youtube comment should be flagged as spam."
            self.CLASSES_TEXT = "1. not spam\n2. spam"

        def format_instruction(self, instruction):
            return f"{instruction}\n{self.CLASSES_TEXT}\n"

        def format_content(self, content):
            return f"comment: {content}\nthe comment is "

    clf = LLMClassifier(model=llm, prompt_formatting=PF())
    loader = DataLoader(samples, batch_size=16, shuffle=False)

    alphas = []
    with torch.no_grad():
        for b in loader:
            alphas.append(clf.soft_labels_batch(b))

    alpha = torch.cat(alphas, dim=0)
    tu, du, ku = compute_uncertainties(alpha)
    torch.save({"total": tu, "data": du, "knowledge": ku},
               f"youtube_uncertainties_epoch{epoch}.pt")


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
        "Student train brier score:", evaluation.compute_metric(gt_labels_train, probs_np, "brier")
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
        optimizer = optim.AdamW(llm_params, lr=learning_rate)

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
        yahoo_alpha_test = evaluate()
        #yahoo_uncertainties(yahoo_alpha_test, epoch)
        #amazon_uncertainties(epoch)
        #sst2_uncertainties(epoch)
        #youtube_uncertainties(epoch)

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



alpha_test = evaluate()
#yahoo_uncertainties(alpha_test,epoch)
#amazon_uncertainties(epoch)
#sst2_uncertainties(epoch)
#youtube_uncertainties(epoch)
train_student()



