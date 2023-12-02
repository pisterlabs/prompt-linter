#load model, make sure it works
from langchain.llms import RWKV
import os
os.environ["RWKV_CUDA_ON"] = '1' # if '1' then use CUDA kernel for seq mode (much faster)

weight_path = r'/models/rwkv-raven-detailed-rolls.pth'
tokenizer_json = r"/20B_tokenizer.json"


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# Instruction:
{instruction}

# Input:
{input}

# Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

# Instruction:
{instruction}

# Response:
"""

model = RWKV(model=weight_path, strategy="cuda fp16i8 *20 -> cuda fp16", tokens_path=tokenizer_json)
response = model(generate_prompt("Write a python code that prints the first 10 numbers of the fibonacci sequence."))