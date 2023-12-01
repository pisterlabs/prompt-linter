#GGUF

#https://github.com/marella/ctransformers

!pip install ctransformers
!pip install ctransformers[cuda]
!pip install langchain

from langchain.llms import CTransformers

config = {'max_new_tokens': 2056, 'repetition_penalty': 1.1, 'gpu_layers': 50}

llm = CTransformers(model='TheBloke/CodeLlama-13B-GGUF', config=config)

llm("write only a code for scrapy crawl type spider to scrape urls using a text document having all the links")


#GGML

!pip install ctransformers ctransformers[cuda]


from ctransformers import AutoModelForCausalLM

#llm = AutoModelForCausalLM.from_pretrained("marella/gpt-2-ggml", model_file="ggml-model.bin")

llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-GGML", gpu_layers=50)

llm("AI is going to")