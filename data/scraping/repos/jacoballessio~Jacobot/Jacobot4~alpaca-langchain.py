from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline, BitsAndBytesConfig

from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

import torch

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
tokenizer = LlamaTokenizer.from_pretrained("chainyo/alpaca-lora-7b")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = LlamaForCausalLM.from_pretrained(
    "chainyo/alpaca-lora-7b",
    load_in_8bit=True,
    torch_dtype=torch.float32,
    quantization_config=quantization_config,
).to(device)
print(base_model)

# Move some layers to the CPU manually
#layers_to_move = list(range(10, 12))
#for i, layer in enumerate(base_model.model.layers):
#    if i in layers_to_move:
#        layer.to("cpu")

pipe = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=tokenizer,
    max_length=256,
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.2
)

local_llm = HuggingFacePipeline(pipeline=pipe)

template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: 
{instruction}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["instruction"])

llm_chain = LLMChain(prompt=prompt,
                     llm=local_llm 
                     )
#get time
import time
start_time = time.time()
print("Started response at %s" % start_time)
question = "What are alpacas? and how are they different from llamas?"

print(llm_chain.run(question))

print("--- %s seconds ---" % (time.time() - start_time))
#get user input

question = input("Ask me a question: ")

print(llm_chain.run(question))
