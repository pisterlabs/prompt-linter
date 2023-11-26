from langchain.llms import ChatGLM
from langchain.tools import DuckDuckGoSearchResults
from TextEmbeddingModels.FAISS import FaissSearch
from Prompt.CustomPromptTemplates import DocumentQAPromptTemplate
from langchain import PromptTemplate, LLMChain
import os

class Robot():
    def __init__(self) -> None:
        self.endpoint_url = os.environ.get('ENDPOINT_URL', 'http://192.168.100.20:8000')

    def init_llm(self):
        llm = ChatGLM(endpoint_url = self.endpoint_url)
        return llm

    def duckduck_search(self,question):
        search = DuckDuckGoSearchResults()
        res = search.run(question)
        return res

Test = Robot()
template = """{question}"""

prompts = PromptTemplate(template=template, input_variables=["question"])

while True:
    question = input("用户：\n")
    print("\n")
    llm = Test.init_llm()
    llm_chain = LLMChain(prompt=prompts,llm=llm)
    
    db = FaissSearch()
    docs = db.search(question,db_local='/home/db/test_combine_db')
    doc = f"""
{docs[0].page_content}
{docs[1].page_content}
{docs[2].page_content}
{docs[3].page_content}
    """
    explainer = DocumentQAPromptTemplate(input_variables=["question"])
    prompt = explainer.format(document=doc,question=question)
    ans = llm_chain.run(prompt)
    print(f"模型：\n{ans}")
    print("\n")
    print("----------------------------")

        


