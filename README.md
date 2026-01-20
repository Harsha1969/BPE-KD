# Uncertainty Modeling in Large Language Models (LLMs)

This project presents a framework for efficient uncertainty estimation in Large Language Models (LLMs) by distilling uncertainty-aware knowledge from **Bayesian teacher models** into a student LLM.

The student model outputs **Dirichlet concentration parameters** instead of softmax probabilities, enabling it to represent both predicted class probabilities and associated uncertainty. Fine-tuning is done using **LoRA** (Low-Rank Adaptation) for memory efficiency.

---

## Approach Overview

### 1. **Teacher Models – Bayesian Uncertainty Estimators**
Bayesian teacher models are used to generate uncertainty-aware predictive distributions.

- **Bayesian Prompt Ensembles (BayesPE)**
  - Multiple semantically equivalent prompts are queried.
  - Prompt weights are learned via variational inference on a small validation set.
  - Final prediction is a weighted combination of all prompt outputs.

- **Laplace-LoRA**
  - A LoRA-finetuned LLM is treated as a Bayesian model using a Laplace approximation.
  - The posterior is approximated around the MAP solution.
  - Predictive uncertainty is obtained by marginalizing over the approximate posterior.

Both teacher models provide calibrated predictive distributions used for student distillation.

### 2. **Student Model**
- Two variants of student models are trained i.e. softmax and dirichlet output students.
- Softmax student is the standard model which outputs probabilities.
- It is trained using **KL divergence loss**.
- Dirichlet student has modified final layer which produce Dirichlet parameters instead of probabilities: `α = 1 + softplus(logits)`.
- Trained using a **Dirichlet-based distillation loss** to match teacher behavior.
- LoRA adapters used for efficient fine-tuning.

### 3. **Single-Pass Inference**
- Once trained, the student can output both predictions and uncertainties using a single forward pass.

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

### Step 1: Run the Teacher Inference
#### 1A. Bayesian Prompt Ensembles (BayesPE)
For each dataset, run the corresponding notebook to:
- Query the model with multiple prompts.
- Save prompt-wise class probabilities and learned weights which are needed to train student models.
- Evaluate performance on the test data.

```bash
# Open and run the notebook
amazon_teacher.ipynb
sst2_teacher.ipynb
yahoo_teacher.ipynb
youtube_teacher.ipynb
```

#### 1B. Laplace LoRA
- Finetune the LLM using LoRA on the datasets and save the checkpoints.
  ```bash
   accelerate launch custom_run_gpt_amazon.py
   accelerate launch custom_run_gpt_sst2.py
   accelerate launch custom_run_gpt_yahoo.py
   accelerate launch custom_run_gpt_youtube.py
  ```
- Run post-hoc Laplace approximation on saved checkpoints.
  ```bash
  accelerate launch custom_run_gpt_amazon_laplace.py
  accelerate launch custom_run_gpt_sst2_laplace.py
  accelerate launch custom_run_gpt_yahoo_laplace.py
  accelerate launch custom_run_gpt_youtube_laplace.py
  ```
### Step 2: Train the Student Models
#### 2A. Softmax-based Student (using KL Divergence Loss)

- Trained using minimization of KL divergence between student and teacher probabilities as objective.
- Evaluate performance on the test data.
  
```bash
python amazon_softmax_student.py
python sst2_softmax_student.py
python yahoo_softmax_student.py
python youtube_softmax_student.py
```
#### 2B. Dirichlet-based Student (using Dirichlet based distillation Loss)

- This student learns to predict Dirichlet parameters that match the teacher's ensembled behavior.
- Train using dirichlet based distillation loss between student and teacher probabilities.
- Evaluate performance on the test data.
- Different variants of regularizers are supported.
#### Command-Line Arguments

This script supports multiple Dirichlet student training modes via command-line arguments.
You can switch between **standard**, **fixed α₀**, and **learnable α₀** setups without modifying the code.


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
  
The files `amazon_teacher.ipynb`, `amazon_softmax_student.py`, and `amazon_dirichlet_student.py` also include **out-of-distribution (OOD) evaluation**.  
These models are trained only on the **Amazon Reviews** dataset and tested on other domains such as **Yahoo Answers**, **SST-2**, and **YouTube Comments** to assess out-of-distribution detection abilities of model and predictive uncertainty.
