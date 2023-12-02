import os
import requests
from pathlib import Path

from langchain.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.vectorstores import Chroma 

from ray import serve
from starlette.requests import Request

# simple example inspired from 
# https://medium.com/geekculture/automating-pdf-interaction-with-langchain-and-chatgpt-e723337f26a6
# and https://github.com/sophiamyang/tutorials-LangChain/tree/main


@serve.deployment(route_prefix="/",
                  autoscaling_config={
                        "min_replicas": 2,
                        "initial_replicas": 2,
                        "max_replicas": 4,
    })
class AnswerPDFQuestions():
    def __init__(self, vector_db_path: str, hf_ai_key: str, verbose=False):

        # Load the embeddings 
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_ai_key
    
        embeddings = HuggingFaceEmbeddings()
        self._vectordb = self._vectordb = Chroma(persist_directory=vector_db_path, 
                                                 embedding_function=embeddings)
        template = """Question: {question}

        Answer: """
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=HuggingFaceHub(repo_id="google/flan-t5-xl",
                                                                model_kwargs={"temperature":0, "max_length":128}))
        # self._pdf_qa_chain = ConversationalRetrievalChain.from_llm(llm_chain, self._vectordb.as_retriever())
        self._pdf_qa_chain = llm_chain
        self._chat_history=[]

    async def __call__(self, http_request: Request) -> str:
        json_request: str = await http_request.json()
        prompts = []
        for prompt in json_request:
            text = prompt["text"]
            if isinstance(text, list):
                prompts.extend(text)
            else:
                prompts.append(text)
        result = self._pdf_qa_chain({"question": prompts[0], "chat_history": self._chat_history})
        
        return result
    
if __name__ == "__main__":

    HF_TOKEN = "you_token""
    document = Path(Path.cwd(), "hai_ai_index_report.pdf").as_posix()
    vector_db_path = Path(Path.cwd(), "vector_hf_db").as_posix()
    deployment = AnswerPDFQuestions.bind(vector_db_path, HF_TOKEN)

    serve.run(deployment)

    # send the request to the deployment
    prompts = [ "What is the total number of publications?",
                "What is the percentage increase in the number of AI-related job postings?",
                "What are the top takeaways from this report?",
                "List benchmarks are released to evaulate AI workloads?",
                "Describe and summarize the techincal ethical issues raised in the report?",
                "Why Chinese citizens are more optimistic about AI than Americans?",
                "How many bills containing “artificial intelligence” were passed into law?",
                "What is the percentage increase in the number of AI-related job postings?"
    ]
    
    sample_inputs = [{"text": prompt} for prompt in prompts]
    for sample_input in sample_inputs:
        output = requests.post("http://localhost:8000/", json=[sample_input]).json()
        print(f"Question: {output['question']}\nAnswer: {output['text']}\n")
        

    serve.shutdown()
