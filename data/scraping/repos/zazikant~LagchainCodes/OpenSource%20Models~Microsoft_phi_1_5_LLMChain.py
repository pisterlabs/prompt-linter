!pip -q install git+https://github.com/huggingface/transformers # need to install from github
!pip install -q datasets loralib sentencepiece
!pip -q install bitsandbytes accelerate xformers einops
!pip -q install langchain

!nvidia-smi

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_default_device('cuda')



model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5",
                                             trust_remote_code=True,
                                             torch_dtype="auto")

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5",
                                          trust_remote_code=True,
                                          torch_dtype="auto")

def generate(input_text, max_new_tokens=500):
    inputs = tokenizer(input_text, return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tokenizer.batch_decode(outputs)[0]

    print(text)
    
    
!pip install transformers
!pip install huggingface_hub
    

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

from transformers import pipeline
pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=256)


#below works without error for passing formatted prompt but cannot run without llmchain
local_llm = HuggingFacePipeline(pipeline=generate(prompt.format(product='colorful socks')))

#or

#local_llm = HuggingFacePipeline(pipeline=generate("write your prompt here"))



chain = LLMChain(llm=local_llm, prompt=prompt, output_key='shashi')