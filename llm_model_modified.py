import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from peft import get_peft_model, LoraConfig, TaskType

class LLM(object):
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3", temperature=0.7, max_tokens=200, use_pipeline=False, use_reduced_precision=True, use_lora=True):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_pipeline = use_pipeline
        self.use_lora = use_lora
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        # Enable mixed precision
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
        """Apply LoRA for efficient fine-tuning without modifying the full model."""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  
            lora_alpha=32, 
            lora_dropout=0.05,
            target_modules=["q_proj","k_proj", "v_proj","o_proj", "lm_head"] 
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
                
    def return_logits(self, instructions, content=None):
        if content is not None:
            prompt_text = f"{instructions}\n{content}"
        else:
            prompt_text = f"{instructions}"
    
        indexed_tokens = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
    
        outputs = self.model(indexed_tokens) 
    
        logits = outputs.logits[0, -1, :]  
    
        return logits

    
    def word_logit(self, logits, word):
          """
          given logits over the vocabulary and a word, it returns the logit for the word.
    
          :param logits: logits over the vocabulary [n_vocabulary].
          :param word: string containing the word to index (if multiple words given, it uses the first one).
    
          :return: logit(word).
          """
          if self.model_name.startswith('mistralai') or self.model_name.startswith('google'):
              idx = self.tokenizer.encode(word)[1]
          else:
              idx = self.tokenizer.encode(word)[0]
          return logits[idx].to(dtype=torch.float32)
        
    def class_probabilities(self, instructions, content, class_words):
        """Compute class probabilities given class keywords."""
        logits = self.return_logits(instructions, content)
        class_logits = torch.zeros(len(class_words), device=logits.device)
        
        for i, word in enumerate(class_words):
            class_logits[i] = self.word_logit(logits, word)
        
        alpha_c = torch.nn.functional.softplus(class_logits) + 1  
        return alpha_c

    def generate_text(self, instructions, content):
        """Generates text using the model."""
        prompt_text = f"{instructions}\n\n{content}\n"

        if self.use_pipeline:
            if not hasattr(self, 'pipeline'):
                self.pipeline = transformers.pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
            sequences = self.pipeline(
                prompt_text,
                num_return_sequences=1,
                max_length=self.max_tokens,
                **self.generate_kwargs
            )
            return sequences[0]['generated_text'][len(prompt_text):]
        
        indexed_tokens = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(indexed_tokens, **self.generate_kwargs)
        
        torch.cuda.empty_cache()  # Free GPU memory after generation
        
        return self.tokenizer.decode(outputs[0])[len(prompt_text):]
