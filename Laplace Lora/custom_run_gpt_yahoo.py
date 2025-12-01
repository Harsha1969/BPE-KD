import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
import copy

import datasets
from datasets import Dataset, DatasetDict
import pandas as pd
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo, login
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    BitsAndBytesConfig,
    LlamaForCausalLM, LlamaTokenizer
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)

# Hugging Face Login
login(token="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")

logger = get_logger(__name__)

# ADDED: PromptFormatting class for Yahoo Answers
class PromptFormatting(object):
    def __init__(self):
        self.INSTRUCTION = 'Identify the topic that the following question and answer belong to:'
        self.CLASSES = [
            'Society & Culture', 'Science & Mathematics', 'Health', 'Education & Reference',
            'Computers & Internet', 'Sports', 'Business & Finance', 'Entertainment & Music',
            'Family & Relationships', 'Politics & Government'
        ]
        self.CHOICE_LETTERS = "ABCDEFGHIJ"
        self.CLASSES_TEXT = "\n".join([f"{self.CHOICE_LETTERS[i]}. {self.CLASSES[i]}" for i in range(len(self.CLASSES))])

    def format_prompt(self, content):
        return f"{self.INSTRUCTION}\n\nChoices:\n{self.CLASSES_TEXT}\n\nText:\n---\n{content}\n---\n\nAnswer:"

prompt_formatter = PromptFormatting()

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default='custom_yahoo', # MODIFIED: Default task
        help="The name of the task to train on.",
    )
    parser.add_argument(
        "--train_file", type=str, default='train_yahoo.csv', help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default='test_yahoo.csv', help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512, # Increased for longer prompts
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
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
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=.5,
        help="Gradient clipping norm.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
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
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
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
    parser.add_argument(
        "--save_train_results",
        action="store_true",
        default=False,
        help="Whether or not to save evaluation on training set.",
    )
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--testing_set", type=str, default='val')
    parser.add_argument("--lm_head", action="store_true", default=False)
    args = parser.parse_args()

    print(args)

    peft_method = 'lora'
    if args.lm_head:
        peft_method = 'lora_lmhead'
    if args.testing_set != 'val':
        peft_method += args.testing_set

    model_name_safe = args.model_name_or_path.replace("/", "_")
    args.output_dir += f'/{args.task_name}/{model_name_safe}_{peft_method}_{args.lora_alpha}_{args.lora_dropout}_{args.learning_rate}_{args.seed}'

    return args


def main():
    args = parse_args()
    send_example_telemetry("run_glue_no_trainer", args)

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

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
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
    else:
        # Fallback for other tasks
        raw_datasets = load_dataset(args.task_name)


    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    print("Loading Model Weights........")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    print("Model loaded........")

    target_modules = ['v_proj', 'q_proj']
    if args.lm_head:
        target_modules.append('lm_head')
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules
    )
    model = get_peft_model(model, peft_config)
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
            
            # The label is 1-10, so subtract 1 to make it 0-9
            result["labels"] = [int(label) - 1 for label in examples["label"]]
        else: # Fallback for other tasks
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
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
    
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.mixed_precision == "fp16" else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

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
            labels = kwargs.pop('labels', None)
            output_dict = self.model(**kwargs)
            logits = output_dict['logits']
            
            if self.id_list:
                selected_logits = logits[:, -1, self.id_list]
                output_dict['logits'] = selected_logits
            
            if labels is not None:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(output_dict['logits'], labels)
                output_dict['loss'] = loss

            return output_dict

    model = WrappedModel(model)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    metric = evaluate.load("accuracy")

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.model.save_pretrained(
                        output_dir,
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                    )
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(output_dir)

            if completed_steps >= args.max_train_steps:
                break
        
        model.eval()
        samples_seen = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataset) - samples_seen]
                    references = references[: len(eval_dataset) - samples_seen]
                else:
                    samples_seen += len(references)
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()