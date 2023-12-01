import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

from langchain import HuggingFaceHub

repo_id = "Writer/camel-5b-hf" # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64})

from langchain import PromptTemplate, LLMChain

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Who won the FIFA World Cup in the year 1994? what was the players of the team that won? "

print(llm_chain.run(question))