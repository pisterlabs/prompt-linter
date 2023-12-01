!pip install transformers==4.32.0 accelerate tiktoken einops scipy transformers_stream_generator==0.0.4 peft deepspeed
!pip install auto-gptq optimum

#Long-Context Understanding (L-CU) is a technique that extends the context length of a language model, such as Qwen-7B-Chat. This is achieved through the use of NTK-aware interpolation and LogN attention scaling. To enable these techniques, set the use_dynamic_ntk and use_logn_attn flags in the config.json file to true. On the VCSUM long-text summary dataset, Qwen-7B-Chat achieved impressive Rouge-L results when using these techniques.

# !git clone -b v1.0.8 https://github.com/Dao-AILab/flash-attention
# %cd flash-attention


import os
import torch
import transformers


from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModel

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat-Int4", trust_remote_code=True)

!nvidia-smi

model = AutoModelForCausalLM.from_pretrained(
     "Qwen/Qwen-VL-Chat-Int4",
     device_map="auto",
     trust_remote_code=True, torch_dtype=torch.bfloat16
 ).eval()

from transformers import pipeline

pipe = pipeline("text-generation",
                 model=model,
                 tokenizer= tokenizer,
                 torch_dtype=torch.float16,
                 device_map="auto",
                 max_new_tokens = 1256,
                 do_sample=True,
                 top_k=30,
                 num_return_sequences=1,
                 eos_token_id=tokenizer.eos_token_id
                 )


!pip install -q langchain

from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain

llm = HuggingFacePipeline(pipeline = pipe)

#llm('write a just a one word answer: which is bigger in size an giraff or dinosaur? \n\n')

#----------------------------------------------below is for qwen vl chat int4----------------------------------------------

query = tokenizer.from_list_format([
    {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
    {'text': 'Generate the caption in English with grounding:'},
])
inputs = tokenizer(query, return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(**inputs)
response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
print(response)
# <img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>Generate the caption in English with grounding:<ref> Woman</ref><box>(451,379),(731,806)</box> and<ref> her dog</ref><box>(219,424),(576,896)</box> playing on the beach<|endoftext|>
image = tokenizer.draw_bbox_on_latest_picture(response)
if image:
  image.save('2.jpg')
else:
  print("no box")
  
  