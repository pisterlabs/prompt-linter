import os
import argparse
import plotly.express as px
from transformers import AutoTokenizer
import datasets
import random
import torch
import numpy as np
import pandas as pd
from langchain import PromptTemplate

from utils import get_logger


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)



class InstructDataset:
    def __init__(self, args):
        self.args = args
        self.logger = get_logger("Dataset logger","info")
        if self.args.token is None:
            self.logger.info("No token passed, looking at HF_TOKEN environment variable")
            self.args.token = os.getenv("HF_TOKEN",None)
            if self.args.token is None:
                self.logger.info("HF_TOKEN not set")

        
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model,trust_remote_code=True,token=args.token,add_eos_token=False,add_bos_token=False)


        self.end_of_text_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token is None:
            self.beginning_of_text_token = self.end_of_text_token
        else:
            self.beginning_of_text_token = self.tokenizer.bos_token

        instruct_prompt_template = self.beginning_of_text_token + "{instruction}\n\nOutput:\n{response}" + self.end_of_text_token
        self.instruct_prompt_template = PromptTemplate(
            input_variables=["instruction","response"],
            template=instruct_prompt_template)
        
        instruct_prompt_template_with_context = self.beginning_of_text_token + "{instruction}\n\nInput:\n{context}\n\nOutput:\n{response}" + self.end_of_text_token
        self.instruct_prompt_template_with_context = PromptTemplate(
            input_variables=["instruction", "context", "response"],
            template=instruct_prompt_template_with_context)

        self.dataset = self.create_instruction_dataset()
        self.train_dataset,self.validation_dataset,self.test_dataset = self.split_dataset()
        self.save_dataset_splits()


    def create_instruction_dataset(self):
        all_entries = self.get_base_instruction_dataset()
        all_lengths = []

        for entry in all_entries:
            tokenized_entry = self.tokenizer(entry, return_tensors="np")
            tokenized_entry_length = len(tokenized_entry["input_ids"][0])
            all_lengths.append(tokenized_entry_length)

        if self.args.histogram:
            fig = px.histogram(all_lengths)
            fig.show()
        print(f"Max length: {max(all_lengths)}")
        print(f"Min length: {min(all_lengths)}")
        print(f"Mean length: {np.mean(all_lengths)}")
        print(f"Median length: {np.median(all_lengths)}")
        print(f"Std: {np.std(all_lengths)}")
        print(f"Number of entries: {len(all_entries)}")
        return all_entries



    def get_base_instruction_dataset(self):
        all_prompts = []

        instruct_dataset = datasets.load_dataset(self.args.dataset)["train"]
        for item in instruct_dataset:
            instruction = item["instruction"]
            context = item["context"]
            response = item["response"]
            if context == "" or context is None:
                prompt = self.instruct_prompt_template.format(instruction=instruction,response=response)
            else:
                prompt = self.instruct_prompt_template_with_context.format(instruction=instruction,context=context,response=response)

            tokenized_entry_input_ids = self.tokenizer(prompt, return_tensors="np")["input_ids"][0]
            tokenized_entry_length = len(tokenized_entry_input_ids)
            if tokenized_entry_length >= self.args.max_tokens:
                self.logger.info("Too long, not adding to dataset")
            elif self.args.pad_token_id is not None and self.args.pad_token_id in tokenized_entry_input_ids:
                self.logger.info("Contains pad token, not adding to dataset")
                self.logger.warn("Recreate the dataset with a different pad token")
                print(prompt)
                quit()
            else:
                all_prompts.append(prompt)
        return all_prompts

    def split_dataset(self,train_size=0.8,validation_size=0.15,test_size=0.05):
        random.shuffle(self.dataset)
        train_size = int(train_size*len(self.dataset))
        validation_size = int(validation_size*len(self.dataset))
        test_size = int(test_size*len(self.dataset))
        train_dataset = self.dataset[:train_size]
        validation_dataset = self.dataset[train_size:train_size+validation_size]
        test_dataset = self.dataset[train_size+validation_size:]
        return train_dataset,validation_dataset,test_dataset

    def save_dataset_splits(self):
        train_name = self.args.name + "_" + "train_" + str(self.args.seed) + ".csv"
        validation_name = self.args.name + "_" + "validation_" + str(self.args.seed) + ".csv"
        test_name = self.args.name + "_" + "test_" + str(self.args.seed) + ".csv"
        train_path = os.path.join(os.path.realpath('./'),train_name)
        validation_path = os.path.join(os.path.realpath('./'),validation_name)
        test_path = os.path.join(os.path.realpath('./'),test_name)

        train_df = pd.DataFrame(self.train_dataset,columns=["text"])
        validation_df = pd.DataFrame(self.validation_dataset,columns=["text"])
        test_df = pd.DataFrame(self.test_dataset,columns=["text"])

        train_df.to_csv(train_path,index=False)
        validation_df.to_csv(validation_path,index=False)
        test_df.to_csv(test_path,index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-s","--seed", type=int, help="Seed for random number generator", default=42)
    parser.add_argument("-m","--model", type=str, help="What model to use for the tokenizer", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("-t","--token",type=str,help="Huggingface token",default=None)
    parser.add_argument("-n","--name",type=str,help="What name to save the csv files under",default="instruct")
    parser.add_argument("-d","--dataset",type=str,help="What huggingface dataset to use",default="databricks/databricks-dolly-15k")
    parser.add_argument("--histogram",action="store_true",default=False)
    parser.add_argument("--max_tokens",type=int,help="Maximum number of tokens to use",default=1024)
    parser.add_argument("--pad_token_id",type=int,default=18610)

    args = parser.parse_args()
    seed_all(args.seed)
    
    instruct_dataset = InstructDataset(args)