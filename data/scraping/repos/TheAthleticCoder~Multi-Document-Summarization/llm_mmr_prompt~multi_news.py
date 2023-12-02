import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from torch import cuda
import torch
from tqdm import tqdm
import pandas as pd
import time
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain
import csv

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate summaries using a language model.")
parser.add_argument("--model_id", type=str, help="Hugging Face model ID")
parser.add_argument("--file_path", type=str, help="Path to the input CSV file")
parser.add_argument("--new_file_save_path", type=str, help="Path to save the generated CSV file")
args = parser.parse_args()

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
print(device)

model_id = args.model_id
hf_auth = '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'
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, use_auth_token=hf_auth)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth)
print(f"Model size: {model.get_memory_footprint():,} bytes")

DEFAULT_SYSTEM_PROMPT = """\
You are an agent who generates a summary using key ideas and concepts. You have to ensure that the summaries are coherent, fluent, relevant and consistent.
"""
instruction = """
"""
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
"""
DO NOT CHANGE ANY OF THE BELOW FUNCTIONS
"""
def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS #system prompt: Default instruction to be given to the model
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST # Final Template: takes in instruction as well. Here it would take in the summary and the source
    return prompt_template
#Function to remove the prompt from the final generated answer
def cut_off_text(text, prompt):
    cutoff_phrase = prompt
    index = text.find(cutoff_phrase)
    if index != -1:
        return text[:index]
    else:
        return text
def remove_substring(string, substring):
    return string.replace(substring, "")
### What torch.autocast is: casts tensors to a smaller memory footprint
def generate(text):
    prompt = get_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs,
                             max_new_tokens=512,
                             eos_token_id=tokenizer.eos_token_id,
                             pad_token_id=tokenizer.eos_token_id,
                             )
    final_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    final_outputs = cut_off_text(final_outputs, '</s>')
    final_outputs = remove_substring(final_outputs, prompt)

    return final_outputs#, outputs
def parse_text(text):
        wrapped_text = textwrap.fill(text, width=100)
        print(wrapped_text +'\n\n')
        # return wrapped_text

from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain
from transformers import pipeline
pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens = 1024,
                #do_sample=True,
                #top_k=30,
                #num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                )
llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':0})

system_prompt = """
You are an agent who is tasked with creating cohesive and relevant text that integrates key ideas and concepts provided in a list format. Your goal is to produce summaries that are fluent, coherent, and consistent.
"""
instruction = """
Generate the required text using the list of key ideas and concepts provided below:\n{text}
"""
#Loading and setting the system prompt
template = get_prompt(instruction, system_prompt)
prompt = PromptTemplate(template=template, input_variables=["text"])
llm_chain = LLMChain(prompt=prompt, llm=llm)
print(template)

df = pd.read_csv(args.file_path)
df['mmr'] = df['mmr'].apply(eval)
# df['mmr_length'] = df['mmr'].apply(len)
# N = 50
# filtered_df = df[df['mmr_length'] > N]
# print(filtered_df.head())

# df = filtered_df
df['generated_text'] = ""  # Adding a new column to the dataframe

with open(args.new_file_save_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)    
    csvwriter.writerow(df.columns)
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Rows"):
        generated_output = llm_chain.run(row['mmr'])
        row['generated_text'] = generated_output
        csvwriter.writerow(row)
    
#python generate_summary.py --model_id "mistralai/Mistral-7B-Instruct-v0.1" --file_path "sample_test_0.2_mmr.csv" --new_file_save_path "sample_test_0.2_mmr_generated.csv"

