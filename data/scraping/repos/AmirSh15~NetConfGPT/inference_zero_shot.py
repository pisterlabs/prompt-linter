import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

from utils.utils import remove_descriptions

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_OUTPUT_LENGTH = 2200
TOP_K = 50
TOP_P = 0.95
TEMPERATURE = 0.1
DO_SAMPLE = True
SAVE_PATH = "/home/amir/NetConfGPT/results/"
access_token = "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"
EXTENDER = "_ZERO_SHOT"

# first purge cuda memory
torch.cuda.empty_cache()

# check if SAVE_PATH exists
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH, exist_ok=True)

# Prepare quantized config
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the model
model_id = "mistralai/Mistral-7B-v0.1" # "mistralai/Mistral-7B-Instruct-v0.1" # "mistralai/Mistral-7B-v0.1" # "bigcode/starcoderbase-1b" # "Salesforce/codegen25-7b-multi" # "cerebras/btlm-3b-8k-base" # "meta-llama/Llama-2-70b-chat-hf" # "tiiuae/falcon-40b"
model_name = model_id.split("/")[-1]
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # load_in_8bit_fp32_cpu_offload=True, # for falcon
    # load_in_4bit=True, # for QLoRA
    quantization_config=nf4_config,
    trust_remote_code=True,
    device_map="auto",
    token=access_token,
)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

config = AutoConfig.from_pretrained(model_id)
model_max_length = config.max_position_embeddings
logger.info(f"max_length: {model_max_length}")

# define pipeline
pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=MAX_OUTPUT_LENGTH,
    model_kwargs={
        "temperature": TEMPERATURE, 
        "top_k": TOP_K, 
        "top_p": TOP_P, 
        # "num_return_sequences": 5, 
        # "max_length":MAX_OUTPUT_LENGTH,
        # "pad_token_id":tokenizer.eos_token_id,
        },
)

llm = HuggingFacePipeline(pipeline=pipe)

##### prepare the prompt

file_name = "visited_networks"
# read xml file
xml_file_path = f"/home/amir/NetConfGPT/examples/{file_name}.xml"
with open(xml_file_path, "r") as f:
    xml_file = f.read()
    
# read yang file
yang_file_path = f"/home/amir/NetConfGPT/examples/{file_name}.yang"
with open(yang_file_path, "r") as f:
    yang_file = f.read()
yang_file = remove_descriptions(yang_file)

# read test yang file
yang_file_path = f"/home/amir/NetConfGPT/examples/ocsset.yang"
with open(yang_file_path, "r") as f:
    yang_file_test = f.read()    
yang_file_test = remove_descriptions(yang_file_test)

template = """Convert the Input YANG to XML format.

YANG: {yang_test}
XML:
"""
prompt = PromptTemplate(template=template, input_variables=["yang_test"])
prompt_len = tokenizer.encode(prompt.format_prompt(yang_test=yang_file_test).text, return_tensors="pt").shape[1]
logger.info(f"Prompt length: {prompt_len}")
######

chain = prompt | llm

# # invoke the chain
# logger.info("Invoking the chain...")
# start_time = time.time()
# output_lc = chain.invoke({"yang_test": yang_file_test})
# end_time = time.time()
# logger.info(f"Chain took {int(end_time - start_time)} seconds")

# # remove the text after "</config>"
# output_lc = output_lc.split("</config>")[0] + "</config>" if "</config>" in output_lc else output_lc

# # save the output in a text file
# save_path = os.path.join(SAVE_PATH, model_name + "_" + f"output_langchain{EXTENDER}.xml")
# with open(save_path, "w") as f:
#     f.write(output_lc)
# logger.info(f"Saved the output in {save_path}")


# inference model
prompt = prompt.format_prompt(yang_test=yang_file_test).text


##### manual inference
input_ids = tokenizer.encode(prompt, return_tensors="pt")
# move to cuda
input_ids = input_ids.to("cuda")

# inference
logger.info("Invoking the model manually...")
start_time = time.time()
output = model.generate(
    input_ids,
    do_sample=DO_SAMPLE,
    max_length=min(prompt_len + MAX_OUTPUT_LENGTH, model_max_length),
    top_k=TOP_K,
    top_p=TOP_P,
    num_return_sequences=1,
    temperature=TEMPERATURE,
    pad_token_id=tokenizer.eos_token_id,
    # repetition_penalty=1.7,
)
# decode
output_text = tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)
end_time = time.time()
logger.info(f"Model took {int(end_time - start_time)} seconds")

# remove the input prompt from the output
output_text = [seq.replace(prompt, "") for seq in output_text]

# remove the text after "</config>"
output_text = [seq.split("</config>")[0] + "</config>" if "</config>" in seq else seq for seq in output_text]

# save the output in a text file
save_path = os.path.join(SAVE_PATH, model_name + "_" + f"output_model{EXTENDER}.xml")
with open(save_path, "w") as f:
    f.write("\n\n\n\n".join(output_text))
logger.info(f"Saved the output in {save_path}")
#####

# # inference using pipeline
# pipe = pipeline(
#     "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=MAX_OUTPUT_LENGTH,
#     model_kwargs={
#         "temperature": TEMPERATURE, 
#         "top_k": TOP_K, 
#         "top_p": TOP_P, 
#         # "num_return_sequences": 5, 
#         # "max_length":MAX_OUTPUT_LENGTH,
#         # "pad_token_id":tokenizer.eos_token_id,
#         },
#     return_full_text=False,
# )
# logger.info("Invoking the pipeline manually...")
# start_time = time.time()
# sequences = pipe(
#     prompt,
#     # max_length=MAX_OUTPUT_LENGTH,
#     do_sample=DO_SAMPLE,
#     top_k=TOP_K,
#     top_p=TOP_P,
#     temperature=TEMPERATURE,
#     num_return_sequences=1,
#     # eos_token_id=tokenizer.eos_token_id,
# )
# end_time = time.time()
# logger.info(f"Pipeline took {int(end_time - start_time)} seconds")

# # save the output in a text file
# seq_text = [seq['generated_text'] for seq in sequences]

# # remove the input prompt from the output
# seq_text = [seq.replace(prompt, "") for seq in seq_text]

# # remove the text after "</config>"
# seq_text = [seq.split("</config>")[0] + "</config>" if "</config>" in seq else seq for seq in seq_text]

# save_path = os.path.join(SAVE_PATH, model_name + "_" + f"output_pipeline{EXTENDER}.xml")
# with open(save_path, "w") as f:
#     f.write("\n\n\n\n".join(seq_text))
# logger.info(f"Saved the output in {save_path}")