from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import os

class Hugging_face:
    def __init__(self, api_token : str):
        self.api_token = api_token
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = self.api_token

    def predict(self, data : str, template : str, repo_id : str = "google/flan-t5-xxl", **kwargs : dict) -> str:
        prompt : PromptTemplate = PromptTemplate(template=template, input_variables=["data"])
        llm : HuggingFaceHub = HuggingFaceHub(
            repo_id=repo_id, model_kwargs=kwargs
        )
        llm_chain : LLMChain = LLMChain(prompt=prompt, llm=llm)
        return llm_chain.run(data)