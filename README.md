# 🔍 Uncertainty Modeling in Large Language Models (LLMs)

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
| YouTube Comments   | Social Media         | 1,100         | 711          |

---

## Project Structure
 ├──train_*.csv / test_*.csv # Preprocessed datasets
 ├── *_teacher.ipynb # BayesPE inference scripts
 ├── *_softmax_student.ipynb # KL-trained student models (softmax)
 ├── *_dirichlet_student.ipynb # Dirichlet-trained student models
 ├── *_probs.pt # Teacher predictions (prompt-wise)
 ├── *_prompt_weights.pt # Learned prompt weights per sample
 ├── evaluation.py # Final evaluation: F1, ECE, entropy, timing
 ├── llm_model.py, llm_classifier.py # Model loaders and wrappers
 ├── bpe.py, ensemble_scaler.py # BayesPE training utils
 ├── constants.py, requirements.txt # Config and dependencies

## 🔧 Setup Instructions

### Install dependencies

```bash
pip install -r requirements.txt
