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

Datasets can be accessed from the link: https://drive.google.com/drive/folders/1dcoBRWcEM9eFrzFYsrh5YXLxXyqOi7gT?usp=sharing
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
python amazon_softmax_student.py
python sst2_softmax_student.py
python yahoo_softmax_student.py
python youtube_softmax_student.py
```
#### 2B. Dirichlet-based Student (using Dirichlet based distillation Loss)

For each dataset, run the corresponding notebook to:
- This student learns to predict Dirichlet parameters that match the teacher's ensembled behavior.
- Train using the best prompt from BayesPE and dirichlet based distillation loss between student and teacher probabilities.
- Evaluate performance on the test data.
#### Command-Line Arguments

This script supports multiple Dirichlet student training modes via command-line arguments.
You can switch between **normal**, **fixed α₀**, and **learnable α₀** setups without modifying the code.


##### `--mode`
**Type:** `string`  
**Choices:** `standard`, `fixed`, `learnable`  
**Default:** `standard`

Selects the training mode for the Dirichlet student:

- **`standard`**  
  Standard Dirichlet distillation with no constraint or regularization on the concentration parameter α₀.

- **`fixed`**  
  Enforces a fixed Dirichlet concentration α₀ for every sample:
  `alpha = alpha * (fixed_alpha0 / alpha0)`

- **`learnable`**  
  Makes α₀ a learnable global parameter with L2 regularization:
  `alpha0_prior = exp(a)`
  `L_reg = beta * (alpha0_i - alpha0_prior)^2`



##### `--epochs`
**Type:** `int`  
**Default:** `50`

Number of training epochs for the student model.



##### `--batch_size`
**Type:** `int`  
**Default:** `1`

Batch size used during training.  



##### `--lr`
**Type:** `float`  
**Default:** `1e-5`

Learning rate for LLM parameters (e.g., LoRA or unfrozen layers).



##### `--fixed_alpha0`
**Type:** `float`  
**Default:** `10.0`  
**Used only when:** `--mode fixed`

Specifies the constant Dirichlet concentration value α₀ applied to all samples.



##### `--lr_alpha0`
**Type:** `float`  
**Default:** `1e-3`  
**Used only when:** `--mode learnable`

Learning rate for the learnable α₀ parameter `a`, where α₀ = exp(a).


##### `--beta`
**Type:** `float`  
**Default:** `1.0`  
**Used only when:** `--mode learnable`

Regularization strength for the learnable α₀ prior.


### Example Usage

```bash
# Standard Dirichlet student
python amazon_dirichlet_student.py

# Fixed alpha0 student
python amazon_dirichlet_student.py --mode fixed --fixed_alpha0 10

# Learnable alpha0 student
python amazon_dirichlet_student.py --mode learnable --lr_alpha0 1e-3 --beta 1.0
```
Make sure the files *_probs.pt and *_prompt_weights.pt exist before training the student which can be accesible from the link: https://drive.google.com/drive/folders/1dcoBRWcEM9eFrzFYsrh5YXLxXyqOi7gT?usp=sharing
  
The notebooks `amazon_teacher.ipynb`, `amazon_softmax_student.py`, and `amazon_dirichlet_student.py` also include **out-of-distribution (OOD) evaluation**.  
These models are trained only on the **Amazon Reviews** dataset but tested on other domains such as **Yahoo Answers**, **SST-2**, and **YouTube Comments** to assess out-of-distribution detection abilities of model and predictive uncertainty.
