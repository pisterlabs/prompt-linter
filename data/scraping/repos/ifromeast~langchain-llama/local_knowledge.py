from typing import Any, List, Dict, Mapping, Optional
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from models.chinese_text_splitter import ChineseTextSplitter
from langchain.vectorstores import FAISS
from models.custom_llm import CustomLLM
import datetime
import torch
from tqdm import tqdm
from langchain import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain

CHUNK_SIZE = 800
CHUNK_OVERLAP = 70
VECTOR_SEARCH_TOP_K = 2
LOCAL_CONTENT = os.path.join(os.path.dirname(__file__), "../docs")
VS_PATH = os.path.join(os.path.dirname(__file__), "../vector_store/FAISS")
PROMPT_TEMPLATE = """Known information:
{context}

Answer users' questions in a concise and professional manner based on the above known information. If you can't get an answer from them, please give the answer you think is most reasonable. The question is:{question}"""


def load_txt_file(filepath):
    loader = TextLoader(filepath, encoding="utf8")
    textsplitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                         chunk_overlap=CHUNK_OVERLAP,
                                         separators=["\n\n", "\n", " ", ""],
                                         length_function=len)
    docs = loader.load_and_split(text_splitter=textsplitter)
    return docs

def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        try:
            from torch.mps import empty_cache
            empty_cache()
        except Exception as e:
            print(e)
            print("If you are using macOS, please update pytorch to 2.0.0 or more")

def load_file(filepath):
    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.lower().endswith(".pdf"):
        loader = UnstructuredFileLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=True)
        docs = loader.load_and_split(textsplitter)
    else:
        docs = load_txt_file(filepath)
    return docs

def get_related_content(related_docs):
    related_content = []
    for doc in related_docs:
        related_content.append(doc.page_content)
    return "\n".join(related_content)

def get_docs_with_score(docs_with_score):
    docs = []
    for doc, score in docs_with_score:
        doc.metadata["score"] = score
        docs.append(doc)
    return docs

# filepath 可以是目录，也可以是文件
def init_knowledge_vector_store(filepath: str or List[str],
                                vs_path: str or os.PathLike = None,
                                embeddings: object = None):
    loaded_files = []
    failed_files = []
    # 单个文件
    if isinstance(filepath, str):
        if not os.path.exists(filepath):
            print(f"{filepath} path does not exist")
            return None
        elif os.path.isfile(filepath):
            file = os.path.split(filepath)[-1]
            try:
                docs = load_file(filepath)
                print(f"{file} loaded successfully!")
                loaded_files.append(filepath)
            except Exception as e:
                print(e)
                print(f"{file} loading failed!")
                return None
        elif os.path.isdir(filepath):
            docs = []
            for file in tqdm(os.listdir(filepath), desc="Loading files: "):
                fullfilepath = os.path.join(filepath, file)

                try:
                    docs += load_file(fullfilepath)
                    loaded_files.append(fullfilepath)
                except Exception as e:
                    failed_files.append(file)

            if len(failed_files) > 0:
                print("Following files cannot be loaded: ")
                for file in failed_files:
                    print(file,end="\n")
    #  文件列表
    else:
        docs = []
        for file in filepath:
            try:
                docs += load_file(file)
                print(f"{file} loaded successfully!")
                loaded_files.append(file)
            except Exception as e:
                print(e)
                print(f"{file} loading failed!")

    if len(docs) > 0:
        print("Finished file loading, Generating vector store...")
        if vs_path and os.path.isdir(vs_path):
            vector_store = FAISS.load_local(vs_path, embeddings)
            vector_store.add_documents(docs)
            torch_gc()
        else:
            if not vs_path:
                vs_path = os.path.join(vs_path, f"""FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}""")
            vector_store = FAISS.from_documents(docs, embeddings)
            torch_gc()

        vector_store.save_local(vs_path)
        print("Vector generation finished! ")
        return vs_path, loaded_files
    else:
        print("File loading failed! ")
        return None, loaded_files



class LocalDocQA:
    filepath: str
    vs_path: str
    load_files: List[str] = []
    top_k: int
    embedding: object
    llm: object
    conversation_with_summary: object
    init: bool = True

    def __init__(self, filepath: str, vs_path: str, embeddings: object,
                       init: bool = True):
        if init:
            vs_path, loaded_files = init_knowledge_vector_store(filepath=LOCAL_CONTENT,
                                                                vs_path=VS_PATH,
                                                                embeddings=embeddings)
        else:
            vs_path = VS_PATH
            loaded_files = []


        self.load_files = loaded_files
        self.vs_path = vs_path
        self.filepath = filepath
        self.embeddings = embeddings
        self.top_k = VECTOR_SEARCH_TOP_K
        self.llm = CustomLLM()
        # self.conversation_with_summary = ConversationChain(llm=self.llm,
        #                                                memory=ConversationSummaryBufferMemory(llm=self.llm,
        #                                                                                       max_token_limit=40),
        #                                                verbose=True)

    def query_knowledge(self, query: str):
        vector_store = FAISS.load_local(self.vs_path, self.embeddings)
        vector_store.chunk_size = CHUNK_SIZE
        related_docs_with_score = vector_store.similarity_search_with_score(query, k = self.top_k)
        related_docs = get_docs_with_score(related_docs_with_score)
        related_content = get_related_content(related_docs)
        return related_content

    def get_knowledge_based_answer(self, query: str):
        related_content = self.query_knowledge(query)
        prompt = PromptTemplate(
            input_variables=["context","question"],
            template=PROMPT_TEMPLATE,
        )
        pmt = prompt.format(context=related_content,
                            question=query)

        # answer=self.conversation_with_summary.predict(input=pmt)
        answer = self.llm(pmt)
        return answer


if __name__ == "__main__":
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    embeddings = HuggingFaceEmbeddings(model_name="./ckpt/all-mpnet-base-v2",
                                   model_kwargs={'device':EMBEDDING_DEVICE})

    qa_doc = LocalDocQA(filepath=LOCAL_CONTENT,
                        vs_path=VS_PATH,
                        embeddings=embeddings,
                        init=True)
    question = "Find some news about Elon Musk"
    related_content = qa_doc.query_knowledge(query=question)
    print(related_content)
    response = qa_doc.get_knowledge_based_answer(question)
    print(response)