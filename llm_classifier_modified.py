import numpy as np
from tqdm import tqdm
import torch

class LLMClassifier(object):
    def __init__(self, model, prompt_formatting, max_len_content=None):
        """
        Initializes the LLMClassifier.

        Args:
            model (LLM): The language model.
            prompt_formatting: Object containing instruction formatting functions.
            max_len_content (int, optional): Maximum length for content truncation.
        """
        self.model = model
        self.classes_strings = prompt_formatting.CLASSES
        self.instruction = prompt_formatting.INSTRUCTION
        self.classes_for_matching = prompt_formatting.CLASSES_FOR_MATCHING
        self.format_instruction = prompt_formatting.format_instruction
        self.format_content = prompt_formatting.format_content
        self.max_len_content = max_len_content

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.model.to(self.device)

    def truncate_text(self, input_text):
        """
        Truncates input text if it exceeds the maximum allowed length.

        Args:
            input_text (str): The input text to be truncated.

        Returns:
            str: Truncated or original text.
        """
        input_text = str(input_text)
        if self.max_len_content is not None and len(input_text.split()) > self.max_len_content:
            return ' '.join(input_text.split()[:self.max_len_content])
        return input_text

    def make_few_shot_instruction(self, instruction, input_examples=None, labels_examples=None):
        """
        Constructs a few-shot learning instruction.

        Args:
            instruction (str): Instruction text.
            input_examples (list): List of example inputs.
            labels_examples (list): List of example labels.

        Returns:
            str: Formatted few-shot learning instruction.
        """
        instruction = self.format_instruction(instruction)
        few_shots_instructions = ''

        for i in range(len(labels_examples)):
            class_i = self.classes_strings[labels_examples[i]]
            input_i = self.format_content(input_examples[i])
            input_i = self.truncate_text(input_i)
            text_i = f"EXAMPLE {i+1}:\n{instruction}\n{input_i}{class_i}\n\n"
            few_shots_instructions += text_i

        few_shots_instructions += f'EXAMPLE {len(labels_examples) + 1}:\n{instruction}'
        return few_shots_instructions

    def print_prompt_example(self, instruction=None, input_text='<TEXT_IN>', input_examples=None, labels_examples=None):
        """
        Prints an example of how a prompt would be structured.

        Args:
            instruction (str, optional): Instruction text.
            input_text (str, optional): Input text.
            input_examples (list, optional): List of example inputs.
            labels_examples (list, optional): List of example labels.
        """
        if instruction is None:
            instruction = self.instruction
        input_text = self.truncate_text(input_text)
        input_text = self.format_content(input_text)

        if input_examples is None or labels_examples is None:
            instruction = self.format_instruction(instruction)
        else:
            instruction = self.make_few_shot_instruction(instruction, input_examples=input_examples, labels_examples=labels_examples)

        prompt_text = f"{instruction}\n{input_text}<LABEL_OUT>"
        print(prompt_text)

    def soft_label(self, instruction=None, input_text='', input_examples=None, labels_examples=None):
        """
        Generates Dirichlet parameters and probabilities for a given input.

        Args:
            instruction (str, optional): Instruction text.
            input_text (str): Input text.
            input_examples (list, optional): Example inputs.
            labels_examples (list, optional): Example labels.

        Returns:
            tuple: (alpha_c, probs) where alpha_c are Dirichlet parameters, and probs are class probabilities.
        """
        if instruction is None:
            instruction = self.instruction

        input_text = self.format_content(input_text)
        
        # Get Dirichlet parameters and class probabilities
        alpha_c = self.model.class_probabilities(instruction, input_text, self.classes_for_matching[0])

        alpha_c = alpha_c.to(self.device)

        return alpha_c

    def soft_labels_batch(self, instruction=None, input_texts='', input_examples=None, labels_examples=None):
        """
        Processes a batch of inputs and returns Dirichlet parameters.

        Args:
            instruction (str, optional): Instruction text.
            input_texts (list): List of input texts.
            input_examples (list, optional): Example inputs.
            labels_examples (list, optional): Example labels.

        Returns:
            tuple: (all_alpha, all_probs) stacked tensors of Dirichlet parameters and probabilities.
        """
        all_alpha, all_probs = [], []
        for i in tqdm(range(len(input_texts))):
            alpha_i = self.soft_label(instruction, input_texts[i], input_examples, labels_examples)
            all_alpha.append(alpha_i)
    
        return torch.stack(all_alpha)
