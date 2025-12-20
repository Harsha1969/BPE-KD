#!/usr/bin/env python
# coding: utf-8

import os
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from llm_classifier_modified import LLMClassifier
from llm_model_modified_Copy2 import LLM
import evaluation


parser = argparse.ArgumentParser()
parser.add_argument("--mode",choices=["standard", "fixed", "learnable"],default="standard",help="Training mode: standard | fixed alpha0 | learnable alpha0")
parser.add_argument("--epochs",type=int,default=50,help="Number of training epochs")
parser.add_argument("--batch_size",type=int,default=32,help="Batch size for training")
parser.add_argument("--lr",type=float,default=1e-6,help="Learning rate for LLM parameters")
parser.add_argument("--fixed_alpha0",type=float,default=10.0,help="Fixed alpha0 value (used when mode=fixed)")
parser.add_argument("--lr_alpha0",type=float,default=1e-3,help="Learning rate for learnable alpha0 parameter a")
parser.add_argument("--beta",type=float,default=1.0,help="Regularization strength for learnable alpha0")
args = parser.parse_args()



# ===============================
# Load Dataset
# ===============================
df_train = pd.read_csv("youtube.csv")

n_train = 1100
n_in_context = 5
n_total_in_context = 9 * n_in_context
n_val = 100

df_train_actual = df_train.iloc[:n_train]
df_test_actual = df_train.iloc[n_train + n_total_in_context + n_val:]

samples_train = df_train_actual.iloc[:, 3].values
gt_labels_train = df_train_actual.iloc[:, 4].values.astype(int)

samples_test = df_test_actual.iloc[:, 3].values
gt_labels_test = df_test_actual.iloc[:, 4].values.astype(int)



class PromptFormatting(object):
    def __init__(self):
        self.INSTRUCTION = 'Judge whether the Youtube comment should be flagged as spam.'
        self.CLASSES = ['not spam', 'spam']
        self.CLASSES_FOR_MATCHING = [
            self.CLASSES,
            ['ham', 'spam'],
            ['0', '1']
        ]
        self.CLASSES_TEXT = '''1. {}\n2. {}'''.format(
            self.CLASSES[0], self.CLASSES[1]
        )

    def format_instruction(self, instruction):
        return '''{}\n{}\n'''.format(instruction, self.CLASSES_TEXT)

    def format_content(self, content):
        return '''comment: {}\nthe comment is '''.format(content)


llm = LLM(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    use_reduced_precision=True,
    use_lora=True
)

prompt_formatting = PromptFormatting()
classifier = LLMClassifier(model=llm, prompt_formatting=prompt_formatting)



probs = torch.load("youtube_llora_teacher_probs.pt", weights_only=False)
weights = torch.full((10000,), 1.0 / 10000.0, dtype=torch.float32)

probs = probs.to(llm.device)
weights = weights.to(llm.device)



class DirichletDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], idx



def dirichlet_loss(alpha, probs, weights):
    alpha_0 = torch.sum(alpha, dim=1, keepdim=True)
    log_gamma_alpha_0 = torch.lgamma(alpha_0)
    log_gamma_alpha_c = torch.lgamma(alpha).sum(dim=1, keepdim=True)

    weighted_log_probs = (alpha.unsqueeze(-1) - 1) * torch.log(probs + 1e-8)
    class_sum = weighted_log_probs.sum(dim=1)

    if weights.ndim == 1:
        weights = weights.unsqueeze(1)

    prompt_sum = (class_sum * weights.T).sum(dim=1, keepdim=True)
    return -(log_gamma_alpha_0 - log_gamma_alpha_c + prompt_sum).mean()

a = torch.nn.Parameter(torch.tensor(2.3025851, device=llm.device))
def alpha0_l2_regularizer(alpha, a, beta):
    alpha0 = alpha.sum(dim=1)
    alpha0_prior = torch.exp(a)
    return beta * ((alpha0 - alpha0_prior) ** 2).mean()


def evaluate():
    def dirichlet_to_prob(alpha):
        return alpha / alpha.sum(dim=1, keepdim=True)

    class TestDirichletDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    llm.model.eval()
    test_dataset = TestDirichletDataset(samples_test)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    all_alpha = []
    with torch.no_grad():
        for batch_samples in test_dataloader:
            alpha_batch = classifier.soft_labels_batch(input_texts=batch_samples)
            all_alpha.append(alpha_batch)

    alpha_test = torch.cat(all_alpha, dim=0)
    stu_probs = dirichlet_to_prob(alpha_test).cpu().numpy()

    f1_score = evaluation.compute_metric(gt_labels_test, stu_probs, metric='f1')
    ece = evaluation.compute_metric(gt_labels_test, stu_probs, metric='ece')
    acc = evaluation.compute_metric(gt_labels_test, stu_probs, metric='acc')
    nll = evaluation.compute_metric(gt_labels_test, stu_probs, metric='nll')
    brier = evaluation.compute_metric(gt_labels_test, stu_probs, metric='brier')

    print(
        'Student test f1-score: {}, Student test ECE: {}, '
        'Student test Accuracy: {}, Student test NLL: {},'
        'Student test brier score: {}'.format(
            f1_score, ece, acc, nll, brier
        )
    )


def evaluate_train(epoch_alpha):
    probs = epoch_alpha / epoch_alpha.sum(dim=1, keepdim=True)
    probs_np = probs.cpu().numpy()

    f1_score = evaluation.compute_metric(gt_labels_train, probs_np, metric='f1')
    ece = evaluation.compute_metric(gt_labels_train, probs_np, metric='ece')
    acc = evaluation.compute_metric(gt_labels_train, probs_np, metric='acc')
    nll = evaluation.compute_metric(gt_labels_train, probs_np, metric='nll')
    brier = evaluation.compute_metric(gt_labels_train, probs_np, metric='brier')

    print(
        'Student train f1-score: {}, Student train ECE: {}, '
        'Student train Accuracy: {}, Student train NLL: {},'
        'Student train brier score: {}'.format(
            f1_score, ece, acc, nll, brier
        )
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

        epoch_alpha = torch.cat(epoch_alphas, dim=0)

        if args.mode == "learnable":
            print(
                f"Epoch {epoch+1}/{args.epochs}, "
                f"Dirichlet Loss: {dirichlet_loss_term}, "
                f"Regularizer Loss: {regularizer_loss}, "
                f"Loss: {total_loss}, "
                f"alpha0_prior: {torch.exp(a).item()}"
            )
        else:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss}")

        evaluate_train(epoch_alpha)
        evaluate()

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



evaluate()
train_student()

