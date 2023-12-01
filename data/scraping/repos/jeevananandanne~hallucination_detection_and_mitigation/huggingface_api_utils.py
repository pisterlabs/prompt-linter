import torch
import torch.nn.functional as F
from examples.few_shot.openai_api_utils import openai_result_to_json
import numpy as np
from os import sys

from fairseq.utils import print_r0
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM
except:
    pass

class HuggingFaceAPIHelper:
    def __init__(self, model_name):
        # if we have changed the model name to avoid / with =, change it back for use with HuggingF ace 
        model_name = model_name.replace("=", "/")
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.build_model(self.config)


    def build_model(self, config):
        architecture = config.architectures[0]
        if "LMHeadModel" in architecture:
            self.model = AutoModelForCausalLM.from_pretrained(config.name_or_path)
            self.arch_type = "causal_lm"
        elif "Seq2SeqLM" in architecture or "ConditionalGeneration" in architecture:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(config.name_or_path)
            self.arch_type = "seq2seq_lm"
        else:
            raise Exception("Could not match any known architecture.")
        
        # Parallelize model
        self.model.parallelize()


    def huggingface_result_to_fairseq_result(self, result):
        prompt, result_tokens, result_logprobs = result
        fairseq_result = {"tokens": result_tokens,
        "score": None,
        "attention": None,
        "alignment": None,
        "positional_scores": result_logprobs}
        fairseq_result["gpt3_response"] = "{}"
        return fairseq_result

    
    def call_huggingface_completion_for_seq2seqlm(self, prompt):
        prompt_input = prompt[0].replace("<mask>", "").strip()
        output = prompt[1]
        output_tokens = self.tokenizer(output).input_ids[:-1] # removing eos token
        input_ids = torch.tensor([self.tokenizer(prompt_input).input_ids]).to(self.torch_device)
        output_ids = torch.tensor([self.tokenizer(output).input_ids]).to(self.torch_device)
        with torch.no_grad():
            outputs = self.model(input_ids, labels=output_ids)

            lprobs = F.log_softmax(outputs.logits[0], dim=-1)
            pindex = torch.transpose(output_ids, 1, 0)
            log_probs = torch.gather(lprobs, 1, pindex).squeeze(1)[:-1]

        return (prompt, output_tokens, log_probs)

    
    def call_huggingface_completion_for_causallm(self, prompt):
        prompt = prompt[0].replace("<mask>", prompt[1])
        print_r0(f"prompt: {prompt}")
        text_tokens = self.tokenizer.tokenize(prompt)
        input_ids = self.tokenizer(prompt).input_ids
        bos_id = [self.tokenizer.bos_token_id]
        input_ids_plus = torch.tensor([bos_id + input_ids]).to(self.torch_device)
        with torch.no_grad():
            outputs = self.model(input_ids_plus, labels=input_ids_plus)

            lprobs = F.log_softmax(outputs.logits[0], dim=-1)
            pindex = torch.transpose(input_ids_plus, 1, 0)
            log_probs = torch.gather(lprobs[:-1], 1, pindex[1:]).squeeze(1)

        return (prompt, text_tokens, log_probs)


    def call_huggingface_completion(self, prompt):
        if self.arch_type == "causal_lm":
            return self.call_huggingface_completion_for_causallm(prompt)
        elif self.arch_type == "seq2seq_lm":
            return self.call_huggingface_completion_for_seq2seqlm(prompt)
        else:
            raise Exception(f"Unknown architecture type: {self.arch_type}!")
