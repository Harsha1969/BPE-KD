#!/usr/bin/env python
# coding: utf-8

# ## Student Model(with softmax output) training and evaluation on Amazon reviews polarity dataset

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
from llm_model_modified1 import LLM
import random
import evaluation


# In[3]:


# Load Amazon reviews polarity train and test data
df_train = pd.read_csv('train_amazon.csv', header=None)
df_test = pd.read_csv('test_amazon.csv', header=None)

n_train = 10000
n_in_context = 5  
n_total_in_context = len(df_train) * n_in_context  
n_test = 5000
n_val = 100

df_train_actual = df_train.iloc[:n_train] 
df_in_context_base = df_train.iloc[n_train:n_train + n_total_in_context]
df_val = df_train.iloc[n_train + n_total_in_context:n_train + n_total_in_context + n_val]
df_test_actual = df_test.iloc[:n_test]  

gt_labels_train = df_train_actual.iloc[:, 0].values.astype(int) 
samples_train = df_train_actual.iloc[:, 2].values 
gt_labels_val = df_val.iloc[:, 0].values.astype(int) 
samples_val = df_val.iloc[:, 2].values 

gt_labels_test = df_test_actual.iloc[:, 0].values.astype(int)
samples_test = df_test_actual.iloc[:, 2].values  


# In[4]:


# Define a prompt formatting class for sentiment classification and initializes an LLM-based classifier
class PromptFormatting(object):
    def __init__(self):
        # Best instruction from BayesPE teacher i.e. instruction with highest weight
        self.INSTRUCTION = 'classify the sentiment of the Amazon review below into one of the following classes:'
        self.CLASSES = ['negative', 'positive']
        self.CLASSES_FOR_MATCHING = [self.CLASSES, ['neg', 'pos'], ['1', '2']]
        self.CLASSES_TEXT = '''1. {}\n2. {}'''.format(self.CLASSES[0], self.CLASSES[1])

    def format_instruction(self, instruction):
        return '''{}\n{}\n'''.format(instruction, self.CLASSES_TEXT)

    def format_content(self, content):
        return '''review: {}\nthe review is '''.format(content)

llm = LLM(model_name="mistralai/Mistral-7B-Instruct-v0.3", use_reduced_precision=True,use_lora=True)
prompt_formatting = PromptFormatting()
classifier = LLMClassifier(model=llm, prompt_formatting=prompt_formatting)


# In[5]:


# Load teacher predictions and weights
probs = torch.load("amazon_llora_teacher_probs.pt", weights_only=False)

print(probs[0])
# Create weights (CPU)
weights = torch.full((10000,), 1.0 / 10000.0, dtype=torch.float32)

# Move everything to GPU at once
#probs = probs.to(llm.device)
weights = weights.to(llm.device)



# In[6]:


def safe_normalize_and_clamp(x):
    # x shape: (B, C) or similar
    x = x.to(dtype=torch.float32)
    x = x.clamp(min=1e-8)                
    x = x / x.sum(dim=1, keepdim=True)  
    return x

def dirichlet_loss(student_probs, target_probs):
    # both expected shape: (B, C)
    student = safe_normalize_and_clamp(student_probs)
    target = safe_normalize_and_clamp(target_probs)
    # now KL between log(student) and target (target is prob)
    return F.kl_div(student.log(), target, reduction='batchmean')


# In[7]:


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
    test_dataset = DirichletDataset(samples_test, n_test)
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

    print('Student test f1-score: {}, Student test ECE: {}, Student test Accuracy: {}, Student test NLL: {},Student test brier score: {}'.format(f1_score, ece,acc,nll,brier))


# In[ ]:


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


# In[9]:


# Train student model with teacher predictions
def train_student(samples_train, probs, weights, num_epochs=200, learning_rate=1e-5, batch_size=32):
    dataset = DirichletDataset(samples_train, len(samples_train))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, llm.model.parameters()), lr=learning_rate)
    llm.model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        epoch_probs = []
        for batch_idx, (batch_samples, batch_indices) in enumerate(dataloader, start=1):
            #batch_indices = batch_indices.to(llm.device)

            batch_probs = probs[batch_indices]
            weights = weights.view(-1)

            batch_probs = batch_probs.to(llm.device)
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
train_student(samples_train, probs, weights, batch_size=16)