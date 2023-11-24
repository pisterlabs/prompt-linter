import environment
import os
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

###google/flan-t5-xl
repo_id = "google/flan-t5-xl" # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options

# # ###Dolly, by DataBricks
# # #See DataBricks organization page for a list of available models.
# repo_id = "databricks/dolly-v2-3b"

# ###Camel, by Writer
# #See Writer’s organization page for a list of available models.
# repo_id = "Writer/camel-5b-hf" # See https://huggingface.co/Writer for other options

# ###StableLM, by Stability AI
# #See Stability AI’s organization page for a list of available models.
# repo_id = "stabilityai/stablelm-tuned-alpha-3b"
# # Others include stabilityai/stablelm-base-alpha-3b
# # as well as 7B parameter versions


llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64})

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "Who won the FIFA World Cup in the year 1994? "
print(llm_chain.run(question))