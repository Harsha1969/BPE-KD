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
from llm_model_modified1_Copy2 import LLM
import evaluation


# In[ ]:


seed = 2


# In[3]:


# Load Yahoo answers train and test data
df_train = pd.read_csv('train_yahoo.csv', header=None)
df_test = pd.read_csv('test_yahoo.csv', header=None)
n_train = 10000  
n_in_context = 5  
n_val = 100
n_test = 5000
df_train_actual = df_train.iloc[:n_train]
df_test_actual = df_test.iloc[:n_test]
def format_prompt(q1, q2, a):
    return "Question: " + q1.astype(str) + " " + q2.astype(str) + "\nAnswer: " + a.astype(str)

gt_labels_train = df_train_actual.iloc[:, 0].values.astype(int)
samples_train = format_prompt(df_train_actual.iloc[:, 1], df_train_actual.iloc[:, 2], df_train_actual.iloc[:, 3]).values
gt_labels_test = df_test_actual.iloc[:, 0].values.astype(int)
samples_test = format_prompt(df_test_actual.iloc[:, 1], df_test_actual.iloc[:, 2], df_test_actual.iloc[:, 3]).values


# In[4]:


# Define a prompt formatting class for sentiment classification and initializes an LLM-based classifier
class PromptFormatting(object):
    def __init__(self):
        # Best instruction from BayesPE teacher i.e. instruction with highest weight
        self.INSTRUCTION = 'Identify the topic that the following question and answer belong to:'
        self.CLASSES = [
    'Society & Culture',
    'Science & Mathematics',
    'Health',
    'Education & Reference',
    'Computers & Internet',
    'Sports',
    'Business & Finance',
    'Entertainment & Music',
    'Family & Relationships',
    'Politics & Government'
]
        self.CLASSES_FOR_MATCHING = [self.CLASSES]
        self.CLASSES_TEXT = '''1. {}\n2. {}\n3. {}\n4. {}\n5. {}\n6. {}\n7. {}\n8. {}\n9. {}\n10. {}'''.format(self.CLASSES[0],self.CLASSES[1], self.CLASSES[2], self.CLASSES[3], self.CLASSES[4], self.CLASSES[5], self.CLASSES[6], self.CLASSES[7], self.CLASSES[8], self.CLASSES[9])
    def format_instruction(self, instruction):
        return '''{}\n{}\n'''.format(instruction, self.CLASSES_TEXT)

    def format_content(self, content):
        return '''{}\nthe topic is '''.format(content)

prompt_formatting = PromptFormatting()
llm = LLM(model_name="mistralai/Mistral-7B-Instruct-v0.3", use_reduced_precision=True,use_lora=True)
classifier = LLMClassifier(model=llm, prompt_formatting=prompt_formatting)


# In[ ]:


import torch

def compute_uncertainties(probs):
    if isinstance(probs, np.ndarray):
        probs = torch.tensor(probs, dtype=torch.float32)
    total_uncertainty = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return total_uncertainty


# In[ ]:


def yahoo_uncertainties(yahoo_probs_test,epoch):
    yahoo_total_uncertainty = compute_uncertainties(yahoo_probs_test)
    torch.save(yahoo_total_uncertainty,f'softmax_yahoo_yahoo_uncertainties_seed{seed}_epoch{epoch}.pt')


# In[ ]:


def amazon_uncertainties(epoch):
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
    
    prompt_formatting = PromptFormatting()
    classifier = LLMClassifier(model=llm, prompt_formatting=prompt_formatting)
      
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
    amazon_total_uncertainty = compute_uncertainties(stu_probs)
    torch.save(amazon_total_uncertainty,f'softmax_yahoo_amazon_uncertainties_seed{seed}_epoch{epoch}.pt')


# In[ ]:


def sst2_uncertainties(epoch):
    df_train = pd.read_csv('train_sst2.csv')
    df_test = pd.read_csv('test_sst2.csv')
    n_train = 10000  
    n_in_context = 5  
    n_total_in_context = 9 * n_in_context  
    n_val=100
    df_train_actual = df_train.iloc[:n_train] 
    df_test_actual = df_test.iloc[:]  
    gt_labels_train = df_train_actual.iloc[:, 2].values.astype(int) 
    samples_train = df_train_actual.iloc[:, 1].values 
    gt_labels_test = df_test_actual.iloc[:, 2].values.astype(int)
    samples_test = df_test_actual.iloc[:, 1].values 

    class PromptFormatting(object):
        def __init__(self):
            self.INSTRUCTION = 'Select the sentiment category that best matches the opinion expressed in the review snippet.'
            self.CLASSES = ['negative', 'positive']
            self.CLASSES_FOR_MATCHING = [self.CLASSES, ['neg', 'pos'], ['1', '2']]
            self.CLASSES_TEXT = '''1. {}\n2. {}'''.format(self.CLASSES[0], self.CLASSES[1])
    
        def format_instruction(self, instruction):
            return '''{}\n{}\n'''.format(instruction, self.CLASSES_TEXT)
    
        def format_content(self, content):
            return '''review: {}\nthe review is '''.format(content)
    
    prompt_formatting = PromptFormatting()
    classifier = LLMClassifier(model=llm, prompt_formatting=prompt_formatting)

    # Evaluate performance of model on sst2 test data
    class DirichletDataset(Dataset):
        def __init__(self, samples, n_samples):
            self.samples = samples
            self.n_samples = n_samples
    
        def __len__(self):
            return self.n_samples
    
        def __getitem__(self, idx):
            return self.samples[idx]
    
    llm.model.eval()
    test_dataset = DirichletDataset(samples_test, 872)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False) 
    
    def get_test_alpha(test_dataloader, classifier):
        all_alpha = []
    
        with torch.no_grad():
            for batch_samples in test_dataloader:
                alpha_batch = classifier.soft_labels_batch(input_texts=batch_samples)
                all_alpha.append(alpha_batch)
    
        return torch.cat(all_alpha, dim=0) 
    
    stu_probs = get_test_alpha(test_dataloader, classifier)
    import evaluation  
    stu_probs=stu_probs.cpu().numpy()
    f1_score = evaluation.compute_metric(gt_labels_test, stu_probs, metric='f1')
    ece = evaluation.compute_metric(gt_labels_test, stu_probs, metric='ece')
    acc = evaluation.compute_metric(gt_labels_test, stu_probs, metric='acc')
    nll = evaluation.compute_metric(gt_labels_test, stu_probs, metric='nll')
    print('Student f1-score: {}, Student ECE: {}, Student Accuracy: {}, Student NLL: {}'.format(f1_score, ece,acc,nll))

    sst2_total_uncertainty = compute_uncertainties(stu_probs)
    torch.save(sst2_total_uncertainty,f'softmax_yahoo_sst2_uncertainties_seed{seed}_epoch{epoch}.pt')


# In[ ]:


def youtube_uncertainties(epoch):
    df_train = pd.read_csv('youtube.csv')
    n_train = 1100  
    n_in_context = 5 
    
    n_total_in_context = 9 * n_in_context  
    n_val=100
    df_train_actual = df_train.iloc[:n_train] 
    df_in_context_base = df_train.iloc[n_train:n_train + n_total_in_context]
    df_val = df_train.iloc[n_train + n_total_in_context:n_train+n_total_in_context+n_val]
    df_test_actual = df_train.iloc[n_train+n_total_in_context+n_val:]  
    
    gt_labels_train = df_train_actual.iloc[:, 4].values.astype(int) 
    samples_train = df_train_actual.iloc[:, 3].values 
    gt_labels_val = df_val.iloc[:, 4].values.astype(int) 
    samples_val = df_val.iloc[:, 3].values 
    gt_labels_test = df_test_actual.iloc[:, 4].values.astype(int)
    samples_test = df_test_actual.iloc[:, 3].values 

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
    
    prompt_formatting = PromptFormatting()
    classifier = LLMClassifier(model=llm, prompt_formatting=prompt_formatting)

    # Evaluate performance of model on youtube comments test data
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
    import evaluation  
    stu_probs=stu_probs.cpu().numpy()
    f1_score = evaluation.compute_metric(gt_labels_test, stu_probs, metric='f1')
    ece = evaluation.compute_metric(gt_labels_test, stu_probs, metric='ece')
    acc = evaluation.compute_metric(gt_labels_test, stu_probs, metric='acc')
    nll = evaluation.compute_metric(gt_labels_test, stu_probs, metric='nll')
    print('Student f1-score: {}, Student ECE: {}, Student Accuracy: {}, Student NLL: {}'.format(f1_score, ece,acc,nll))

    youtube_total_uncertainty = compute_uncertainties(stu_probs)
    torch.save(youtube_total_uncertainty, f'softmax_yahoo_youtube_uncertainties_seed{seed}_epoch{epoch}.pt')


# In[5]:


# Load teacher predictions and weights
probs = torch.load("yahoo_llora_teacher_probs.pt", weights_only=False)
print(probs.shape)
print(probs[0])
# Create weights (CPU)
weights = torch.full((10000,), 1.0 / 10000.0, dtype=torch.float32)

# Move everything to GPU at once
#probs = probs.to(llm.device)
weights = weights.to(llm.device)

# In[6]:


# Compute KL divergence loss
def dirichlet_loss(student_probs, probs):
    kl_loss = F.kl_div(student_probs.log(), probs, reduction='batchmean')
    return kl_loss


# In[7]:


# Evaluate performance of model on yahoo answers test data
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
    return stu_probs


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
def train_student(samples_train, probs, weights, num_epochs=50, learning_rate=1e-5, batch_size=32):
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

            #if batch_idx % 1000 == 0:
            #        print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}")

        torch.cuda.empty_cache()
        epoch_prob = torch.cat(epoch_probs, dim=0)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss}")
        evaluate_train(epoch_prob)
        stu_probs = evaluate()
        #yahoo_uncertainties(stu_probs,epoch)
        #amazon_uncertainties(epoch)
        #sst2_uncertainties(epoch)
        #youtube_uncertainties(epoch)
    final_train_probs = []
    full_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    llm.model.eval()
    with torch.no_grad():
        for batch_samples, _ in full_dataloader:
            probs = classifier.soft_labels_batch(input_texts=batch_samples)
            final_train_probs.append(probs)
    final_train_probs = torch.cat(final_train_probs, dim=0)
    evaluate_train(final_train_probs)

stu_probs = evaluate()
#yahoo_uncertainties(stu_probs,"pretrained")
#amazon_uncertainties("pretrained")
#sst2_uncertainties("pretrained")
#youtube_uncertainties("pretrained")
train_student(samples_train, probs, weights, batch_size=1)

