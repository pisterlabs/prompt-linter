from langchain.document_loaders import JSONLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()


class KnowledgeBase:
    def create_knowledge_base(self):
        print("calling knowledge_base_init")
        loader = JSONLoader(
            file_path="./resource/flex_message.jsonl",
            jq_schema='"question: "+.question + " answer: " +.answer',
            json_lines=True,
        )

        documents = loader.load()
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPEN-API-KEY"))
        vec_db = FAISS.from_documents(documents, embeddings)
        return vec_db
