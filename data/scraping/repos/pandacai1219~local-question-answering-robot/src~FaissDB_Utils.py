import os
from langchain import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.document_loaders import UnstructuredPDFLoader
from keys import OpenAI_API_KEY
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from template import prompt_template_Document, prompt_template_GPT
import pickle
from logger_config import logger
from langchain.document_loaders import BSHTMLLoader
from langchain.document_loaders.csv_loader import CSVLoader

class FaissDB_Utils:
    def __init__(self, api_key=None,prompt_template=None,temperature=None,max_context_tokens=None,max_response_tokens=None,file_chunk_size=None):
        if api_key is None:
            self.api_key = OpenAI_API_KEY
        else:
            self.api_key = api_key
        if file_chunk_size is None:
            self.file_chunk_size = 200
        else:
            self.file_chunk_size = file_chunk_size
        if temperature is None:
            self.temperature = 0
        else:
            self.temperature = temperature
        if max_context_tokens is None:
            self.max_context_tokens = -1
        else:
            self.max_context_tokens = max_context_tokens
        if prompt_template is None:
            self.prompt_template = prompt_template_Document
        else:
            self.prompt_template = prompt_template
        logger.info(f"self.prompt_template：{self.prompt_template}")
        logger.info(f"self.temperature：{self.temperature}")
        logger.info(f"self.max_context_tokens：{self.max_context_tokens}")
        logger.info(f"self.file_chunk_size：{self.file_chunk_size}")
        logger.info(f"self.api_key：{self.api_key}")
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.text_splitter = CharacterTextSplitter(chunk_size=self.file_chunk_size, chunk_overlap=0)
        self.llm = OpenAI(temperature=self.temperature, max_tokens=self.max_context_tokens, openai_api_key=self.api_key)
        PROMPT = PromptTemplate(template=self.prompt_template, input_variables=["context", "question"])
        logger.info(str(PROMPT))
        self.chain = load_qa_chain(llm=self.llm, chain_type='stuff', verbose=True, prompt=PROMPT)
        self.docCount = 0

    def create_or_import_to_db(self, file_path, filename=None,userName=None):
        db = None
        folder_path="dbf/"+userName
        logger.info(f"folder_path：{folder_path}")
        # 根据文件类型加载文档
        if file_path.endswith(".docx") or file_path.endswith(".doc"):
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_path.endswith(".pdf"):
            loader = UnstructuredPDFLoader(file_path)
        elif filename.endswith(".html"):
            loader = BSHTMLLoader(file_path)
        elif filename.endswith(".csv"):
            loader = CSVLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        documents = loader.load()
        logger.info(str(documents))
        docs = self.text_splitter.split_documents(documents)

        logger.info(f"Found {len(docs)} documents in {file_path}")
        self.docCount=len(docs)
        db_local=None
        try:
            db_local = FAISS.load_local(index_name=userName, embeddings=self.embeddings, folder_path=folder_path)

            for doc in docs:
                doc_temp = [doc]
                db_temp = FAISS.from_documents(documents=doc_temp, embedding=self.embeddings)
                db_local.merge_from(db_temp)

            db_local.save_local(folder_path=folder_path, index_name=userName)
        except Exception as e:
            db = FAISS.from_documents(documents=docs, embedding=self.embeddings)
            db.save_local(folder_path=folder_path, index_name=userName)
            logger.error(f"Error loading db: {e}")
        logger.info(f"Saved db to {filename}{userName}")


    #批量处理目录下的文件     
    def path_to_db(self, directory_path,userName=None):
        db = None
        folder_path="dbf/"+userName
        logger.info(f"folder_path：{folder_path}")
        data = []
        try:
            for filename in os.listdir(directory_path):
                logger.info(f"filename：{filename}")
                if filename.endswith(".docx") or filename.endswith(".doc"):
                    loader = UnstructuredWordDocumentLoader(f'{directory_path}/{filename}')
                    data.append(loader.load())
                elif filename.endswith(".txt"):
                    loader = TextLoader(f'{directory_path}/{filename}', encoding='utf-8')
                    data.append(loader.load())
                elif filename.endswith(".pdf"):
                    loader = UnstructuredPDFLoader(f'{directory_path}/{filename}')
                    data.append(loader.load())
                elif filename.endswith(".html"):
                    loader = BSHTMLLoader(f'{directory_path}/{filename}')
                    data.append(loader.load())
                elif filename.endswith(".csv"):
                    loader = CSVLoader(f'{directory_path}/{filename}')
                    data.append(loader.load())
                else:
                    continue
            print(len(data))
        except Exception as e:
            print(f"Error loading documents: {e}")

        text = []
        for i in range(len(data)):
            text.append(self.text_splitter.split_documents(data[i]))
        print(len(text))

        db_local=None
        try:
            db_local = FAISS.load_local(index_name=userName, embeddings=self.embeddings, folder_path=folder_path)

            for doc in text:
                doc_temp = [doc]
                db_temp = FAISS.from_documents(documents=doc_temp, embedding=self.embeddings)
                db_local.merge_from(db_temp)

            db_local.save_local(folder_path=folder_path, index_name=userName)
        except Exception as e:
            for doc in text:
                #如果是第一次循环，对db_local进行初始化
                if db_local is None:
                    db_local = FAISS.from_documents(documents=doc, embedding=self.embeddings)
                else:
                    db_local.merge_from(FAISS.from_documents(documents=doc, embedding=self.embeddings))
            db_local.save_local(folder_path=folder_path, index_name=userName)
            logger.error(f"Error loading db: {e}")
        logger.info(f"Saved db to {directory_path}{userName}")
    

    def search_documents(self, query, userName=None):
        folder_path="dbf/"+userName
        logger.info(f"folder_path：{folder_path}")

        db = FAISS.load_local(index_name=userName,embeddings=self.embeddings,folder_path=folder_path)

        results  = db.similarity_search(query, k=3)

        return results