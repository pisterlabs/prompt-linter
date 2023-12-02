from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain

model_id = "meta-llama/Llama-2-7b-chat-hf"
# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Load Model 
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir='./workspace/', 
    torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", offload_folder="offload")
# Set PT model to inference mode
model.eval()
# Build HF Transformers pipeline 
pipeline = transformers.pipeline(
    "text-generation", 
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_length=400,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

# Test out the pipeline
pipeline('who is kim kardashian?')
# Setup prompt template
template = PromptTemplate(input_variables=['input'], template='{input}') 
# Pass hugging face pipeline to langchain class
llm = HuggingFacePipeline(pipeline=pipeline) 
# Build stacked LLM chain i.e. prompt-formatting + LLM
chain = LLMChain(llm=llm, prompt=template)

response = chain.run('who is kim kardashian?')