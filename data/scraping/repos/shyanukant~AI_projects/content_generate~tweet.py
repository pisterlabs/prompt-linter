## Creating a LLM using LangChain + HuggingFace
import os
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate

hf_key = os.environ.get("SQL_MODEL_KEY")

# model -> https://huggingface.co/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5 = use for hashtag genrator
# model -> https://huggingface.co/tiiuae/falcon-7b-instruct

hub_llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct", 
    model_kwargs = {'temperature' : 0.8, 'max_length':250} ,
    huggingfacehub_api_token=hf_key)

prompt = PromptTemplate(
    input_variables= ['topic'],
    # template= "<|prompter|>Can you write a tweet on {topic}  and encourages engagement from followers.<|endoftext|><|assistant|>" 
    template= "Can you write a tweet on {topic}  and encourages engagement from followers. Use vibrant visuals and witty captions to create excitement around the {topic} and give followers a reason to share and tag their friends."
)
chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)
print(chain.run("how to write clean code"))