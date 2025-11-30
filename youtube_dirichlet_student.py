#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[2]:


import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from llm_classifier_modified import LLMClassifier
from llm_model_modified_Copy2 import LLM
import evaluation


# In[4]:


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


# In[5]:


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


# In[7]:


probs = torch.load("youtube_llora_teacher_probs.pt", weights_only=False)
print(probs[0])
weights = torch.full((10000,), 1.0 / 10000.0, dtype=torch.float32)
probs = probs.to(llm.device)
weights = weights.to(llm.device)

# In[8]:


import torch

def dirichlet_loss(alpha, probs, weights):

    alpha_0 = torch.sum(alpha, dim=1, keepdim=True)                      
    log_gamma_alpha_0 = torch.lgamma(alpha_0)                           
    log_gamma_alpha_c = torch.lgamma(alpha).sum(dim=1, keepdim=True)   

    alpha_expanded = alpha.unsqueeze(-1)                                

    weighted_log_probs = (alpha_expanded - 1) * torch.log(probs + 1e-8) 

    class_sum = weighted_log_probs.sum(dim=1)                           

    if weights.ndim == 1:
        weights = weights.unsqueeze(1)                                   
    weights_broadcasted = weights.T.expand(probs.shape[0], -1)          
    
    weighted_terms = class_sum * weights_broadcasted                  

    prompt_sum = weighted_terms.sum(dim=1, keepdim=True)               

    loss = -(log_gamma_alpha_0 - log_gamma_alpha_c + prompt_sum).mean()

    return loss


# In[ ]:


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
    test_dataset = TestDirichletDataset(samples_test, 711)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False) 
    
    def get_test_alpha(test_dataloader, classifier):
        all_alpha = []
    
        with torch.no_grad():
            for batch_samples in test_dataloader:
                alpha_batch = classifier.soft_labels_batch(input_texts=batch_samples)
                all_alpha.append(alpha_batch)
    
        return torch.cat(all_alpha, dim=0) 
    
    alpha_test = get_test_alpha(test_dataloader, classifier)
    stu_probs = dirichlet_to_prob(alpha_test)
    stu_probs=stu_probs.cpu().numpy()
    f1_score = evaluation.compute_metric(gt_labels_test, stu_probs, metric='f1')
    ece = evaluation.compute_metric(gt_labels_test, stu_probs, metric='ece')
    acc = evaluation.compute_metric(gt_labels_test, stu_probs, metric='acc')
    nll = evaluation.compute_metric(gt_labels_test, stu_probs, metric='nll')
    brier = evaluation.compute_metric(gt_labels_test, stu_probs, metric='brier')

    print('Student test f1-score: {}, Student test ECE: {}, Student test Accuracy: {}, Student test NLL: {},Student test brier score: {}'.format(f1_score, ece,acc,nll,brier))


# In[ ]:


def evaluate_train(epoch_alpha):
    def dirichlet_to_prob(alpha):
        return alpha / alpha.sum(dim=1, keepdim=True)

    probs = dirichlet_to_prob(epoch_alpha)
    probs_np = probs.cpu().numpy()

    f1_score = evaluation.compute_metric(gt_labels_train, probs_np, metric='f1')
    ece = evaluation.compute_metric(gt_labels_train, probs_np, metric='ece')
    acc = evaluation.compute_metric(gt_labels_train, probs_np, metric='acc')
    nll = evaluation.compute_metric(gt_labels_train, probs_np, metric='nll')
    brier = evaluation.compute_metric(gt_labels_train, probs_np, metric='brier')

    print('Student train f1-score: {}, Student train ECE: {}, Student train Accuracy: {}, Student train NLL: {},Student train brier score: {}'.format(f1_score, ece,acc,nll,brier))


# In[9]:


from torch.utils.data import Dataset, DataLoader

class DirichletDataset(Dataset):
    def __init__(self, samples, num_samples):
        self.samples = samples
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples[idx], idx 


# In[10]:


# Train student model with teacher predictions and evaluate after each epoch on test data
def train_student(samples_train, probs, weights, num_epochs=50, learning_rate=1e-5,batch_size=32):
    dataset = DirichletDataset(samples_train, len(samples_train))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, llm.model.parameters()), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        llm.model.train()
        epoch_alphas = []
        for batch_idx, (batch_samples, batch_indices) in enumerate(dataloader, start=1):
            batch_indices = batch_indices.to(llm.device)

            batch_probs = probs[batch_indices] 

            optimizer.zero_grad()

            alpha = classifier.soft_labels_batch(input_texts=batch_samples)
            alpha = torch.clamp(alpha, min=1e-3)
            epoch_alphas.append(alpha.detach().cpu()) 
            loss = dirichlet_loss(alpha, batch_probs, weights)
          
            loss.backward()

            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 1000 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}")

        torch.cuda.empty_cache()
        epoch_alpha = torch.cat(epoch_alphas, dim=0)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss}")
        evaluate_train(epoch_alpha)
        evaluate()
    
    final_train_alphas = []
    full_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    llm.model.eval()
    with torch.no_grad():
        for batch_samples, _ in full_dataloader:
            alpha = classifier.soft_labels_batch(input_texts=batch_samples)
            final_train_alphas.append(alpha)
    final_train_alphas = torch.cat(final_train_alphas, dim=0)
    evaluate_train(final_train_alphas)

evaluate()
train_student(samples_train, probs, weights, batch_size=32)