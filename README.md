# Uncertainty Modeling in Large Language Models (LLMs)

This project presents a framework for efficient uncertainty estimation in Large Language Models (LLMs) by distilling the knowledge from a Bayesian Prompt Ensemble (BayesPE) into a student LLM. The approach allows reliable uncertainty estimation in a **single forward pass**, overcoming the computational inefficiencies of traditional Bayesian inference methods.

The student model outputs Dirichlet concentration parameters instead of softmax probabilities, enabling it to represent both predicted class probabilities and the associated confidence. Fine-tuning is done using **LoRA** (Low-Rank Adaptation) for memory efficiency.

---

## Approach Overview

### 1. **Teacher Model – Bayesian Prompt Ensembles (BayesPE)**
- Multiple semantically equivalent prompts are queried.
- Prompt weights are learned via variational inference on a small validation set.
- Final prediction is a weighted combination of all prompt outputs.

### 2. **Student Model – Dirichlet Output LLM**
- Final layer modified to produce Dirichlet parameters: `α = 1 + softplus(logits)`.
- Trained using a **Dirichlet-based distillation loss** to match teacher behavior.
- LoRA adapters used for efficient fine-tuning.

### 3. **Single-Pass Inference**
- Once trained, the student can output both probabilities and uncertainties using a single forward pass.

---

## Datasets Used

| Dataset            | Domain               | Train Samples | Test Samples |
|--------------------|----------------------|---------------|--------------|
| Amazon Reviews     | Sentiment Analysis   | 10,000        | 5,000        |
| SST-2              | Sentiment Analysis   | 10,000        | 872          |
| Yahoo Answers      | Topic Classification | 10,000        | 5,000        |
| YouTube Comments   | Spam Detection       | 1,100         | 711          |

## Setup Instructions

### Install dependencies

```bash
pip install -r requirements.txt
```
## How to Run

All scripts use **Mistral-7B-Instruct v0.3** as the base model and require GPU (A100 recommended ~40GB).

### Step 1: Run the Teacher (BayesPE) Inference

For each dataset, run the corresponding notebook to:
- Query the model with multiple prompts.
- Save prompt-wise class probabilities and learned weights.
- Evaluate performance on the test data.

```bash
# Open and run the notebook
amazon_teacher.ipynb
sst2_teacher.ipynb
yahoo_teacher.ipynb
youtube_teacher.ipynb
```
### Step 2: Train the Student Models
#### 2A. Softmax-based Student (using KL Divergence Loss)

For each dataset, run the corresponding notebook to:
- Train using the best prompt from BayesPE and KL divergence between student and teacher probabilities.
- Evaluate performance on the test data.
  
```bash
# Open and run the notebook
amazon_softmax_student.ipynb
sst2_softmax_student.ipynb
yahoo_softmax_student.ipynb
youtube_softmax_student.ipynb
```
#### 2B. Dirichlet-based Student (using Weighted Dirichlet Likelihood Loss)

For each dataset, run the corresponding notebook to:
- This student learns to predict Dirichlet parameters that match the teacher's ensembled behavior.
- Train using the best prompt from BayesPE and dirichlet based distillation loss between student and teacher probabilities.
- Evaluate performance on the test data.

```bash
# Open and run the notebook
amazon_dirichlet_student.ipynb
sst2_dirichlet_student.ipynb
yahoo_dirichlet_student.ipynb
youtube_dirichlet_student.ipynb
```
Make sure the files *_probs.pt and *_prompt_weights.pt exist before training the student.
