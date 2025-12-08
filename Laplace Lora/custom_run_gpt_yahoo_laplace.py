import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    LlamaForCausalLM, LlamaTokenizer,
    BitsAndBytesConfig
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
import pandas as pd
import csv
from dataclasses import dataclass
import numpy as np
import evaluation 

from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PeftModel,
    PeftConfig
)
from laplace import Laplace
import pickle
import dill

logger = get_logger(__name__)

# ADDED: Your PromptFormatting class, adapted for multiple-choice
class PromptFormatting(object):
    def __init__(self):
        self.INSTRUCTION = 'Identify the topic that the following question and answer belong to:'
        self.CLASSES = [
            'Society & Culture', 'Science & Mathematics', 'Health', 'Education & Reference',
            'Computers & Internet', 'Sports', 'Business & Finance', 'Entertainment & Music',
            'Family & Relationships', 'Politics & Government'
        ]
        self.CHOICE_LETTERS = "ABCDEFGHIJ"
        # Format as A. Class1, B. Class2, etc.
        self.CLASSES_TEXT = "\n".join([f"{self.CHOICE_LETTERS[i]}. {self.CLASSES[i]}" for i in range(len(self.CLASSES))])

    def format_prompt(self, content):
        # Combines instruction, choices, and content into a single prompt
        return f"{self.INSTRUCTION}\n\nChoices:\n{self.CLASSES_TEXT}\n\nText:\n---\n{content}\n---\n\nAnswer:"

prompt_formatter = PromptFormatting()


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default='custom_yahoo',
        help="The name of the task to train on.",
    )
    parser.add_argument(
        "--train_file", type=str, default='/net/storage/pr3/plgrid/plggploraxs/plgharsha/BPE-KD/BPE-KD/train_yahoo.csv', help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default='/net/storage/pr3/plgrid/plggploraxs/plgharsha/BPE-KD/BPE-KD/test_yahoo.csv', help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512, # Increased max length for the longer prompt
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    # ... (rest of parse_args is unchanged)
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='mistralai/Mistral-7B-Instruct-v0.3',
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=10000,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default='./outputs', help="Where to store the final model.")
    parser.add_argument("--peft_method", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default='1000',
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        default=True,
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--load_step", type=int, default=9999)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--laplace_hessian", type=str, default='kron')
    parser.add_argument("--laplace_sub", type=str, default='all')
    parser.add_argument("--laplace_prior", type=str, default='homo', help='homo')
    parser.add_argument("--laplace_optim_step", type=int, default=1000)
    parser.add_argument("--testing_set", type=str, default='val')
    parser.add_argument("--laplace_predict", type=str, default='mc_corr', help='probit bridge bridge_norm mc_indep mc_corr')
    parser.add_argument("--lm_head", action="store_true", default=False)
    args = parser.parse_args()
    return args


def main(load_step):
    args = parse_args()
    # ... (rest of main function setup is unchanged)
    args.load_step = load_step
    send_example_telemetry("run_glue_no_trainer", args)

    peft_method = 'lora'
    if args.lm_head:
        peft_method = 'lora_lmhead'
    if args.testing_set != 'val':
        peft_method += args.testing_set

    model_name_safe = args.model_name_or_path.replace("/", "_")
    args.output_dir += f'/{args.task_name}/{model_name_safe}_{peft_method}_{args.lora_alpha}_{args.lora_dropout}_{args.learning_rate}_{args.seed}'
    args.laplace_output_dir = f'outputs_laplace/{args.task_name}/{model_name_safe}_{peft_method}_{args.lora_alpha}_{args.lora_dropout}_{args.learning_rate}_{args.seed}/'
    
    laplace_output_dir = args.laplace_output_dir + f'step_{args.load_step}'
    os.makedirs(laplace_output_dir, exist_ok=True)

    accelerator = (
        Accelerator(log_with=args.report_to, project_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.task_name == 'custom_yahoo':
        if not args.train_file or not args.validation_file:
            raise ValueError("--train_file and --validation_file must be provided for custom_yahoo")
        
        # Load local CSVs with no header
        df_train_full = pd.read_csv(args.train_file, header=None)
        df_test_full = pd.read_csv(args.validation_file, header=None)
        
        # Assign column names based on the dataset structure
        df_train_full.columns = ['label', 'question_title', 'question_content', 'best_answer']
        df_test_full.columns = ['label', 'question_title', 'question_content', 'best_answer']
        
        # Slice to desired sizes
        train_df = df_train_full.iloc[:10000]
        validation_df = df_test_full.iloc[:5000]
        
        raw_datasets = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(validation_df)
        })

    else: # Fallback to other datasets
        raw_datasets = load_dataset(args.task_name)


    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    final_output_dir = args.output_dir
    peft_config = PeftConfig.from_pretrained(final_output_dir)
    
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, final_output_dir)
    model.print_trainable_parameters()

    for name, param in model.named_parameters():
        param.requires_grad = False
        if 'lora' in name:
            if args.laplace_sub == 'all':
                param.requires_grad = True

    model.print_trainable_parameters()

    padding = "max_length" if args.pad_to_max_length else False
    
    def preprocess_function(examples):
        if args.task_name == 'custom_yahoo':
            # Combine text columns into a single content string
            contents = []
            for i in range(len(examples['question_title'])):
                title = str(examples['question_title'][i])
                question = str(examples['question_content'][i])
                answer = str(examples['best_answer'][i])
                contents.append(f"Question: {title} {question}\nAnswer: {answer}")
            
            # Format the full prompt using the class
            texts = [prompt_formatter.format_prompt(content) for content in contents]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            
            # The label is already an integer
            result["labels"] = [int(label) for label in examples["label"]] # Subtract 1 to make it 0-9
        else:
            texts = examples["sentence"]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )
    
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.mixed_precision == "fp16" else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    
    if args.testing_set != 'val':
        val_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    class WrappedModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            if args.task_name == 'custom_yahoo':
                # The model should predict one of the letters A, B, C, ..., J
                self.id_list = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in "ABCDEFGHIJ"]
            else:
                self.id_list = None
            self.model = model

        def forward(self, **kwargs):
            model_kwargs = {k: v for k, v in kwargs.items() if k in ["input_ids", "attention_mask"]}
            output_dict = self.model(**model_kwargs)
            logits = output_dict['logits']
            if self.id_list:
                 selected_logits = logits[:, -1, self.id_list]
            else:
                 selected_logits = logits
            return selected_logits.to(torch.float32)

    model = WrappedModel(model)

    # ... (rest of the script is unchanged and omitted for brevity)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],"weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    if args.testing_set == 'val':
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    else:
        model, optimizer, train_dataloader, val_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, eval_dataloader, lr_scheduler
        )
    model.eval()
    metric = evaluate.load("accuracy")
    la = Laplace(model, 'classification',
                 subset_of_weights=args.laplace_sub,
                 hessian_structure=args.laplace_hessian)
    print('----fitting Laplace-----')
    la.fit(train_dataloader)
    if args.testing_set == 'val':
        prior_precision = la.optimize_prior_precision(method='marglik', n_steps=args.laplace_optim_step, lr=1e-1)
        print(f'prior precision: {prior_precision}')
    else:
        prior_precision = la.optimize_prior_precision(method='val_gd', val_loader=val_dataloader, n_steps=args.laplace_optim_step, lr=1e-1)
    torch.save(prior_precision, f'{laplace_output_dir}/prior_precision_{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}_{args.laplace_optim_step}.pt')
    samples_seen = 0
    output_dicts = []
    f_mu_list = []
    f_var_list = []
    all_teacher_samples = []
    all_individual_probs_list = []
    all_probs = []
    all_labels = []
    for step, batch in tqdm(enumerate(train_dataloader)):
        with torch.no_grad():
            f_mu, f_var = la._glm_predictive_distribution(batch)
            f_mu_list.append(f_mu)
            f_var_list.append(f_var)
        samples = 10000
        f_mu_exp = f_mu.expand(samples, -1, -1)
        f_var_exp = f_var.expand(samples, -1, -1, -1)
        jitter = torch.eye(f_var_exp.shape[-1]).to(f_var_exp.device) * 1e-6
        chol = torch.linalg.cholesky(f_var_exp + jitter).to(f_mu_exp.dtype)
        eps = torch.randn_like(f_mu_exp).unsqueeze(-1).to(f_mu_exp.dtype).to(accelerator.device)
        logits_samples = f_mu_exp + (chol @ eps).squeeze(-1)
        probs_samples = torch.softmax(logits_samples, dim=-1)
        probs_to_save = probs_samples.permute(1, 0, 2).cpu()
        all_individual_probs_list.append(probs_to_save)
        probs_samples_permuted = probs_samples.permute(1, 2, 0).cpu()
        all_teacher_samples.append(probs_samples_permuted)
        probs_mean = probs_samples.mean(0)
        all_probs.append(probs_mean.cpu())
        all_labels.append(batch["labels"].cpu())
        predictions = probs_mean.argmax(dim=-1)
        logits = probs_mean.detach()
        for j in range(logits.size(0)):
            probs = logits[j]
            label = batch["labels"]
            output_dict = {
                'index': args.per_device_eval_batch_size * step + j,
                'true': label[j].item(),
                'pred': logits[j].argmax().item(),
                'conf': probs.max().item(),
                'logits': logits[j].cpu().numpy().tolist(),
                'probs': probs.cpu().numpy().tolist(),
            }
            output_dicts.append(output_dict)
        predictions, references = accelerator.gather((predictions, batch["labels"]))
        if accelerator.num_processes > 1:
            if step == len(eval_dataloader) - 1:
                predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                references = references[: len(eval_dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
    f_mu = torch.cat(f_mu_list, dim=0)
    f_var = torch.cat(f_var_list, dim=0)
    torch.save(f_mu, f'{laplace_output_dir}/f_mu_{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}_{args.laplace_optim_step}.pt')
    torch.save(f_var, f'{laplace_output_dir}/f_var_{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}_{args.laplace_optim_step}.pt')
    output_path = os.path.join(laplace_output_dir, f'eval_res_la_{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}_{args.laplace_predict}_{args.laplace_optim_step}.json')
    with open(output_path, 'w+') as f:
        for output_dict in output_dicts:
            f.write(json.dumps(output_dict) + '\n')
    eval_metric = metric.compute()
    all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_probs_np = all_probs.cpu().numpy()
    all_labels_np = all_labels.cpu().numpy()
    try:
        ece = evaluation.compute_metric(all_labels_np, all_probs_np, metric='ece')
        nll = evaluation.compute_metric(all_labels_np, all_probs_np, metric='nll')
        brier = evaluation.compute_metric(all_labels_np, all_probs_np, metric='brier')
        acc = evaluation.compute_metric(all_labels_np, all_probs_np, metric='acc')
        all_results['eval_ece'] = ece
        all_results['eval_nll'] = nll
        all_results['eval_brier'] = brier
        all_results['eval_accuracy_custom'] = acc
        probs_savename = f"{args.task_name}_laplace_probs_{args.load_step}.pt"
        np.save(os.path.join(laplace_output_dir, probs_savename), all_probs_np)
        print(f"Writing probabilities to '{os.path.join(laplace_output_dir, probs_savename)}'")
    except NameError:
        logger.warning("Could not compute custom metrics. Make sure 'evaluation.py' is imported.")
    except Exception as e:
        logger.error(f"An error occurred during custom metric calculation: {e}")
    print("\n--- Final Evaluation Metrics ---")
    for metric_name, value in all_results.items():
        if isinstance(value, (int, float)):
            print(f"{metric_name}: {value:.4f}")
        else:
            print(f"{metric_name}: {value}")
    print("------------------------------------")
    all_results_path = os.path.join(laplace_output_dir, f"all_results_la_{args.laplace_hessian}_{args.laplace_sub}_{args.laplace_prior}_{args.laplace_predict}_{args.laplace_optim_step}.json")
    #with open(all_results_path, "w") as f:
    #    json.dump(all_results, f)
    all_teacher_samples = torch.cat(all_teacher_samples, dim=0)
    teacher_path = os.path.join(laplace_output_dir, "youtube_llora_teacher_probs.pt")
    torch.save(all_teacher_samples, teacher_path)
    print(f"Teacher samples saved to {teacher_path}, shape={all_teacher_samples.shape}")
    #all_individual_probs_tensor = torch.cat(all_individual_probs_list, dim=0)
    #individual_probs_path = os.path.join(laplace_output_dir, "individual_probs.pt")
    #torch.save(all_individual_probs_tensor, individual_probs_path)
    #print(f"Individual probabilities saved to {individual_probs_path}, shape={all_individual_probs_tensor.shape}")
    del model, train_dataloader, la, f_mu, f_var, f_mu_list, f_var_list, metric, eval_metric, output_dicts, eval_dataloader, all_individual_probs_tensor
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main(9999)
