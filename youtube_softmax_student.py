#!/usr/bin/env python
# coding: utf-8


# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# In[2]:


import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from llm_classifier_modified import LLMClassifier
from llm_model_modified1 import LLM
import evaluation


# In[3]:


# Load modified datasets
df_train = pd.read_csv('youtube.csv')
# df_train = df_train.sample(frac=1).reset_index(drop=True)
n_train = 1100  
n_in_context = 5 

n_total_in_context = 9 * n_in_context  
n_val=100
# **Split Data**
df_train_actual = df_train.iloc[:n_train] 
df_in_context_base = df_train.iloc[n_train:n_train + n_total_in_context]
df_val = df_train.iloc[n_train + n_total_in_context:n_train+n_total_in_context+n_val]
df_test_actual = df_train.iloc[n_train+n_total_in_context+n_val:]  

# **Extract Training Data**
gt_labels_train = df_train_actual.iloc[:, 4].values.astype(int) 
samples_train = df_train_actual.iloc[:, 3].values 
gt_labels_val = df_val.iloc[:, 4].values.astype(int) 
samples_val = df_val.iloc[:, 3].values 
# **Extract Test Data (Now from `df_test`)**
gt_labels_test = df_test_actual.iloc[:, 4].values.astype(int)
samples_test = df_test_actual.iloc[:, 3].values 


# In[4]:


# Prompt Formatting Class for YouTube Comment Spam Detection
class PromptFormatting(object):
    def __init__(self):
        self.INSTRUCTION = 'Judge whether the Youtube comment should be flagged as spam.'
        self.CLASSES = ['not spam', 'spam']
        self.CLASSES_FOR_MATCHING = [self.CLASSES, ['ham', 'spam'], ['0', '1']]
        self.CLASSES_TEXT = '''1. {}\n2. {}'''.format(self.CLASSES[0], self.CLASSES[1])

    def format_instruction(self, instruction):
        return '''{}\n{}\n'''.format(instruction, self.CLASSES_TEXT)

    def format_content(self, content):
        return '''comment: {}\nthe comment is '''.format(content)

llm = LLM(model_name="mistralai/Mistral-7B-Instruct-v0.3", use_reduced_precision=True,use_lora=True)
prompt_formatting = PromptFormatting()
classifier = LLMClassifier(model=llm, prompt_formatting=prompt_formatting)


# In[6]:


probs = torch.load("youtube_llora_teacher_probs.pt", weights_only=False)
print(probs.shape)
print(probs[0])
weights = torch.full((10000,), 1.0 / 10000.0, dtype=torch.float32)

# Move everything to GPU at once
probs = probs.to(llm.device)
weights = weights.to(llm.device)



import torch

def dirichlet_loss(student_probs, probs):

    kl_loss = F.kl_div(student_probs.log(), probs, reduction='batchmean')
    return kl_loss


# In[ ]:


def evaluate():
      
    class DirichletDataset(Dataset):
        def __init__(self, samples, n_samples):
            self.samples = samples
            self.n_samples = n_samples
    
        def __len__(self):
            return self.n_samples
    
        def __getitem__(self, idx):
            return self.samples[idx]

    llm.model.eval()
    test_dataset = DirichletDataset(samples_test, 711)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False) 
    
    def get_test_alpha(test_dataloader, classifier):
        all_alpha = []
    
        with torch.no_grad():
            for batch_samples in test_dataloader:
                alpha_batch = classifier.soft_labels_batch(input_texts=batch_samples)
                all_alpha.append(alpha_batch)
    
        return torch.cat(all_alpha, dim=0) 



    stu_probs = get_test_alpha(test_dataloader, classifier)
    stu_probs=stu_probs.cpu().numpy()
    f1_score = evaluation.compute_metric(gt_labels_test, stu_probs, metric='f1')
    ece = evaluation.compute_metric(gt_labels_test, stu_probs, metric='ece')
    acc = evaluation.compute_metric(gt_labels_test, stu_probs, metric='acc')
    nll = evaluation.compute_metric(gt_labels_test, stu_probs, metric='nll')
    brier = evaluation.compute_metric(gt_labels_test, stu_probs, metric='brier')
    print('Student f1-score: {}, Student ECE: {}, Student Accuracy: {}, Student NLL: {},Student brier score: {}'.format(f1_score, ece,acc,nll,brier))


def evaluate_train(epoch_probs):
    probs_np = epoch_probs.cpu().numpy()
    f1_score = evaluation.compute_metric(gt_labels_train, probs_np, metric='f1')
    ece = evaluation.compute_metric(gt_labels_train, probs_np, metric='ece')
    acc = evaluation.compute_metric(gt_labels_train, probs_np, metric='acc')
    nll = evaluation.compute_metric(gt_labels_train, probs_np, metric='nll')
    brier = evaluation.compute_metric(gt_labels_train, probs_np, metric='brier')

    print('Student train f1-score: {}, Student train ECE: {}, Student train Accuracy: {}, Student train NLL: {},Student train brier score: {}'.format(f1_score, ece,acc,nll,brier))


# In[8]:




from torch.utils.data import Dataset, DataLoader

class DirichletDataset(Dataset):
    def __init__(self, samples, num_samples):
        self.samples = samples
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples[idx], idx 


# Train student model with teacher predictions
def train_student(samples_train, probs, weights, num_epochs=50, learning_rate=1e-5, batch_size=32):
    dataset = DirichletDataset(samples_train, len(samples_train))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, llm.model.parameters()), lr=learning_rate)
    llm.model.train()  

    for epoch in range(num_epochs):
        total_loss = 0
        epoch_probs = []
        for batch_idx, (batch_samples, batch_indices) in enumerate(dataloader, start=1):
            batch_indices = batch_indices.to(llm.device)

            batch_probs = probs[batch_indices] 
            weights = weights.view(-1)

                  
            batch_probs = (batch_probs * weights) 
            batch_probs = batch_probs.sum(dim=2) 
            optimizer.zero_grad()

            student_probs = classifier.soft_labels_batch(input_texts=batch_samples)
            epoch_probs.append(student_probs.detach().cpu())
            loss = dirichlet_loss(student_probs, batch_probs)
          
            loss.backward()

            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 1000 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}")

        torch.cuda.empty_cache()
        epoch_prob = torch.cat(epoch_probs, dim=0)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss}")
        evaluate_train(epoch_prob)
        evaluate()
    final_train_probs = []
    full_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    llm.model.eval()
    with torch.no_grad():
        for batch_samples, _ in full_dataloader:
            probs = classifier.soft_labels_batch(input_texts=batch_samples)
            final_train_probs.append(probs)
    final_train_probs = torch.cat(final_train_probs, dim=0)
    evaluate_train(final_train_probs)

evaluate()
train_student(samples_train, probs, weights, batch_size=32)
