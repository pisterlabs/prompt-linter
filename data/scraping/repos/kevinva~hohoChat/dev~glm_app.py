import time
import os, sys

langchain_ChatGLM_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "langchain-ChatGLM")
sys.path.append(langchain_ChatGLM_path)

from configs import model_config
from models.chatglm_llm_old import ChatGLM
from chains.local_doc_qa import similarity_search_with_score_by_vector, LLM_HISTORY_LEN

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredPDFLoader


LOG_PREFIX = "[GLM]"
VECTOR_STORE_PATH = "/root/autodl-tmp/outputs/vector_stores/GLM_FAISS_20230519110529"
DOCS_DATA_DIR = "/root/autodl-tmp/data/track2/"
LLM_MODEL_PATH = "/root/autodl-tmp/models/chatglm-6b/"
EMBEDDING_MODEL_PATH = "/root/autodl-tmp/models/multi-qa-mpnet-base-dot-v1"



def time_str_YmdHMS():
    current_time = time.time()
    local_time = time.localtime(current_time)
    time_str = time.strftime('%Y%m%d%H%M%S', local_time)
    time
    return time_str

def get_filepaths_at_path(item_path):
    if os.path.isfile(item_path):
        return [item_path]
    
    result_list = []
    for item in os.listdir(item_path):
        path = os.path.join(item_path, item)
        file_paths = get_filepaths_at_path(path)
        result_list.extend(file_paths)
    
    return result_list



def init_vector_store(vs_path = None, docs_path = None, embedding_model_path = None):
    start_time = time.time()

    embedding_model_name = os.path.basename(embedding_model_path)
    embeddings = HuggingFaceEmbeddings(model_name = embedding_model_path, 
                                       model_kwargs = {'device': model_config.EMBEDDING_DEVICE})


    if vs_path is not None:
        vector_store = FAISS.load_local(vs_path, embeddings)
        print(f"{LOG_PREFIX} vector_store loaded from {vs_path} successfully! Elapsed time: {time.time() - start_time} seconds")
        return vector_store
    
    if docs_path is None:
        print(f"{LOG_PREFIX} docs_path is None!")
        return None

    file_paths = get_filepaths_at_path(docs_path)
    file_paths = [file_path for file_path in file_paths if file_path.split('.')[-1] == "pdf"]

    print(f"{LOG_PREFIX} Nmber of file_paths: {len(file_paths)}")

    text_splitter = CharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 200)

    docs = []
    for file_path in file_paths:
        loader = UnstructuredPDFLoader(file_path)
        docs += loader.load_and_split(text_splitter)

    print(f"{LOG_PREFIX} Number of docs: {len(docs)}")
    
    vector_store = FAISS.from_documents(docs, embeddings)

    vs_path = f"/root/autodl-tmp/outputs/vector_stores/GLM_FAISS_{embedding_model_name}_{time_str_YmdHMS()}"
    vector_store.save_local(vs_path)

    print(f"{LOG_PREFIX} vector_store saved to {vs_path}")

    FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
    vector_store.chunk_size = model_config.CHUNK_SIZE

    print(f"{LOG_PREFIX} Initial vector_store successfully! Elapsed time: {time.time() - start_time} seconds")

    return vector_store



def init_llm(local_path = None):
    start_time = time.time()

    llm = ChatGLM()

    if local_path is not None:
        llm.load_model(model_name_or_path = local_path,
                    llm_device = model_config.LLM_DEVICE,
                    use_ptuning_v2 = model_config.USE_PTUNING_V2)
    else:
        llm.load_model(model_name_or_path = model_config.llm_model_dict[model_config.LLM_MODEL],
                llm_device = model_config.LLM_DEVICE,
                use_ptuning_v2 = model_config.USE_PTUNING_V2)
    llm.history_len = LLM_HISTORY_LEN

    print(f"{LOG_PREFIX} Initial llm successfully! Elapsed time: {time.time() - start_time} seconds")

    return llm


g_vector_store = init_vector_store(vs_path = None, docs_path = DOCS_DATA_DIR, embedding_model_path = EMBEDDING_MODEL_PATH)
g_llm = init_llm(local_path = LLM_MODEL_PATH)