import os
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

# MODIFIED: Added your PromptFormatting class
class PromptFormatting(object):
    def __init__(self):
        self.INSTRUCTION = 'Is the following Youtube comment spam?'
        self.CLASSES = ['not spam', 'spam']
        self.CLASSES_TEXT = '''1. {}\n2. {}'''.format(self.CLASSES[0], self.CLASSES[1])

    def format_instruction(self, instruction):
        return '''{}\n{}\n'''.format(instruction, self.CLASSES_TEXT)

    def format_content(self, content):
        return '''comment: {}\nthe comment is '''.format(content)

prompt_formatter = PromptFormatting()

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default='custom_youtube', # MODIFIED: Set default to your new task
        help="The name of the task to train on, e.g., 'custom_youtube'.",
    )
    parser.add_argument(
        "--train_file", type=str, default='youtube.csv', help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default='youtube.csv', help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=400,
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

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

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
    
    # MODIFIED: Added pandas-based loading for your custom_youtube task
    if args.task_name == 'custom_youtube':
        if not args.train_file or not args.validation_file:
            raise ValueError("--train_file and --validation_file must be provided for the custom task")
        
        # Load using pandas, which automatically handles headers
        train_df = pd.read_csv(args.train_file)
        validation_df = pd.read_csv(args.validation_file)
	train_df = train_df.iloc[:1100]
        validation_df = validation_df.iloc[1245:]
        # Convert to Hugging Face DatasetDict
        raw_datasets = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(validation_df)
        })
    elif args.task_name == 'custom_task':
        if not args.train_file or not args.validation_file:
            raise ValueError("--train_file and --validation_file must be provided for custom_task")
        data_files = {"train": args.train_file, "validation": args.validation_file}
        raw_datasets = load_dataset("csv", data_files=data_files)
    elif args.task_name in ['wnli', 'rte', 'mrpc', 'cola', 'sst2', 'qnli', 'qqp', 'mnli']:
        raw_datasets = load_dataset("glue", args.task_name, trust_remote_code=True)
    elif args.task_name in ['cb', 'wic', 'boolq']:
        raw_datasets = load_dataset("super_glue", args.task_name)
    elif 'ARC' in args.task_name:
        raw_datasets = load_dataset('ai2_arc', args.task_name)
    elif 'winogrande' in args.task_name:
        raw_datasets = load_dataset("winogrande", args.task_name, trust_remote_code=True)
    else:
        if args.task_name is not None:
             raw_datasets = load_dataset(args.task_name)
        else:
            data_files = {}
            if args.train_file is not None:
                data_files["train"] = args.train_file
            if args.validation_file is not None:
                data_files["validation"] = args.validation_file
            extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
            raw_datasets = load_dataset(extension, data_files=data_files)


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
        if args.task_name == 'custom_youtube':
            # MODIFIED: Use the new prompt formatting class
            instruction = prompt_formatter.format_instruction(prompt_formatter.INSTRUCTION)
            texts = [instruction + prompt_formatter.format_content(comment) for comment in examples["content"]]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            
            map_dict = {False: 0, True: 1}
            result["labels"] = [map_dict[label] for label in examples["class"]]
        elif args.task_name == 'custom_task':
            texts = [f"Please classify the following text as either 'spam' or 'ham'.\n\nText: {text}\nClassification:" for text in examples["text"]]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            
            map_dict = {"ham": 0, "spam": 1}
            result["labels"] = [map_dict[label] for label in examples["label"]]
        elif args.task_name == 'boolq':
            texts = [f"Answer the question with only True or False: {question} Context: {passage}" for passage, question in zip(examples['passage'], examples['question'])]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            result["labels"] = examples["label"]
        elif 'openbookqa' in args.task_name:
            choices_list = [' '.join(f'{label}. {text}' for label, text in zip(choices['label'], choices['text'])) for choices in examples['choices']]
            texts = [f"Select one of the choices that answers the following question: {question} Choices: {choices} Answer:" for question, choices in zip(examples['question_stem'], choices_list)]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            map_dict = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
            result["labels"] = [map_dict[label] for label in examples["answerKey"]]
        elif 'ARC' in args.task_name:
            choices_list = [' '.join(f'{label}. {text}' for label, text in zip(choices['label'], choices['text'])) for choices in examples['choices']]
            texts = [f"Select one of the choices that answers the following question: {question} Choices: {choices} Answer:" for question, choices in zip(examples['question'], choices_list)]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            map_dict = {"A": 0, "B": 1, "C": 2, "D": 3, "1": 0, "2": 1, "3": 2, "4": 3}
            result["labels"] = [map_dict[label] for label in examples["answerKey"]]
        elif 'winogrande' in args.task_name:
            texts = [f"Select one of the choices that answers the following question: {question} Choices: A. {option1}. B {option2}. Answer:" for question, option1, option2 in zip(examples['sentence'], examples['option1'], examples['option2'])]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            map_dict = {"1": 0, "2": 1, "": None}
            result["labels"] = [map_dict[label] for label in examples["answer"]]
        elif args.task_name == 'sst2':
            texts = [f"Classify the sentiment of the following movie review as either positive or negative.\n\nReview: {sentence}\nSentiment:" for sentence in examples["sentence"]]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            result["labels"] = examples["label"]
        else:
            texts = examples["sentence"]
            result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)
            result["labels"] = examples["label"]
        return result

    if args.task_name == "sst2":
        raw_datasets["train"] = raw_datasets["train"].select(range(10000))

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
            # MODIFIED: Added logic for your custom_youtube task
            if args.task_name == 'custom_youtube':
                # 3. DEFINE YOUR TARGET LABEL WORDS HERE.
                # The order must match the mapping: 0 for 'legitimate', 1 for 'spam'
                legitimate_token_id = tokenizer.encode("not spam", add_special_tokens=False)[0]
                spam_token_id = tokenizer.encode("spam", add_special_tokens=False)[0]
                self.id_list = [legitimate_token_id, spam_token_id]
            elif args.task_name == 'custom_task':
                ham_token_id = tokenizer.encode("ham", add_special_tokens=False)[0]
                spam_token_id = tokenizer.encode("spam", add_special_tokens=False)[0]
                self.id_list = [ham_token_id, spam_token_id] # Order must match the integer mapping (0, 1)
            elif args.task_name == 'boolq':
                self.id_list = [tokenizer.encode('False', add_special_tokens=False)[0], tokenizer.encode('True', add_special_tokens=False)[0]]
            elif args.task_name == 'openbookqa':
                self.id_list = [tokenizer.encode('A', add_special_tokens=False)[0], tokenizer.encode('B', add_special_tokens=False)[0], tokenizer.encode('C', add_special_tokens=False)[0], tokenizer.encode('D', add_special_tokens=False)[0]]
            elif 'ARC' in args.task_name:
                self.id_list = [tokenizer.encode('A', add_special_tokens=False)[0], tokenizer.encode('B', add_special_tokens=False)[0], tokenizer.encode('C', add_special_tokens=False)[0], tokenizer.encode('D', add_special_tokens=False)[0]]
            elif 'winogrande' in args.task_name:
                self.id_list = [tokenizer.encode('A', add_special_tokens=False)[0], tokenizer.encode('B', add_special_tokens=False)[0]]
            elif args.task_name == 'sst2':
                neg_token_id = tokenizer.encode("negative", add_special_tokens=False)[0]
                pos_token_id = tokenizer.encode("positive", add_special_tokens=False)[0]
                self.id_list = [neg_token_id, pos_token_id]
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