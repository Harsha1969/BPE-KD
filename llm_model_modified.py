import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from peft import get_peft_model, LoraConfig, TaskType

class LLM(object):
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3", temperature=0.7, max_tokens=200, use_pipeline=False, use_reduced_precision=False, use_lora=True):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_pipeline = use_pipeline
        self.use_lora = use_lora  # Enable LoRA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        model_dtype = torch.float16 if use_reduced_precision and torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, trust_remote_code=True, torch_dtype=model_dtype, device_map="auto"
        )

        if self.use_lora:
            self.apply_lora()

        self.generate_kwargs = {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "use_cache": True,
            "do_sample": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

    def apply_lora(self):
        """Apply LoRA for efficient fine-tuning without modifying full model."""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # LoRA rank
            lora_alpha=32,  # Scaling factor
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"]  # Apply only to attention layers
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def return_logits(self, instructions, content=None):
        """Returns the logits of the model for given instructions and content."""
        prompt_text = f"{instructions}\n{content}" if content else f"{instructions}"
        indexed_tokens = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)

        outputs = self.model(indexed_tokens)
        return outputs.logits[0, -1, :].to(self.device)

    def word_logit(self, logits, word):
        """Extracts the logit for a specific word token."""
        token_ids = self.tokenizer.encode(word, add_special_tokens=False)
        if len(token_ids) > 1:
            raise ValueError(f"Word '{word}' is tokenized into multiple tokens: {token_ids}")
        return logits[token_ids[0]].to(self.device)

    def class_probabilities(self, instructions, content, class_words):
        """Returns Dirichlet parameters and probabilities."""
        logits = self.return_logits(instructions, content)

        class_logits = torch.tensor(
            [self.word_logit(logits, word) for word in class_words], 
            device=self.device, dtype=torch.float32, requires_grad=True
        )

        # Convert logits into Dirichlet parameters
        alpha_c = torch.nn.functional.softplus(class_logits) + 1  
        probs = torch.nn.functional.softmax(class_logits, dim=0)

        return alpha_c, probs

    def generate_text(self, instructions, content):
        """Generates text using the model."""
        prompt_text = f"{instructions}\n\n{content}\n"
        
        if self.use_pipeline:
            sequences = self.pipeline(
                prompt_text,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                max_length=self.max_tokens,
                **self.generate_kwargs
            )
            return sequences[0]['generated_text'][len(prompt_text):]
        
        indexed_tokens = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.module.generate(indexed_tokens, **self.generate_kwargs) if isinstance(self.model, nn.DataParallel) else self.model.generate(indexed_tokens, **self.generate_kwargs)
        return self.tokenizer.decode(outputs[0])[len(prompt_text):]
