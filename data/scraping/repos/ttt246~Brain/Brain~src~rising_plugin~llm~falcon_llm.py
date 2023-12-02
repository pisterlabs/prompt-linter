"""falcon llm"""
from langchain import HuggingFaceHub, PromptTemplate, LLMChain

from Brain.src.common.utils import HUGGINGFACEHUB_API_TOKEN

repo_id = "tiiuae/falcon-7b-instruct"
template = """
You are an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

{question}

"""


class FalconLLM:
    def __init__(self, temperature: float = 0.6, max_new_tokens: int = 2000):
        self.llm = HuggingFaceHub(
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            repo_id=repo_id,
            model_kwargs={"temperature": temperature, "max_new_tokens": max_new_tokens},
        )

    def get_llm(self):
        return self.llm

    def get_chain(self):
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=self.llm, verbose=True)
        return llm_chain

    """getting the output in query with falcon llm"""

    def query(self, question: str) -> str:
        chain = self.get_chain()
        return chain.run(question=question)
