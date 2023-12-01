from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import OpenAI
from llama_index.callbacks import CallbackManager, TokenCountingHandler
import tiktoken
from llama_index.callbacks import CallbackManager, TokenCountingHandler
from llama_index import  ServiceContext
from llama_index import set_global_service_context
def  get_service_context(llm_model):
    #llm = OpenAI(temperature=0.1, model=llm_model)
    llm = OpenAI(model=llm_model)
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model(llm_model).encode,
        verbose=False  # set to true to see usage printed to the console
    )

    callback_manager = CallbackManager([token_counter])
    service_context = ServiceContext.from_defaults(llm=llm,callback_manager=callback_manager)
    return service_context,token_counter

def build_index(llm_model,folderpath,storage_context):
    service_context, token_counter = get_service_context(llm_model)
    documents = SimpleDirectoryReader(folderpath, recursive=True).load_data()
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context,service_context = service_context)
    return index,token_counter
def load_index(llm_model, vector_store):
    service_context, token_counter = get_service_context(llm_model)
    loaded_index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)
    return loaded_index,token_counter

# create a function called index. This function will load the index if storage folder is defined. otherwise it will create the index
def get_index(folderpath,vector_store, load_from_storage=True):
    if load_from_storage:
        print("loading from storage")
        # loaded_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        # return loaded_index
        return load_index(vector_store)
    print("building folder path")
    return build_index(folderpath)
# get index from vector store
def get_index_from_vector_store(llm_model,vector_store):
    print("loading from storage")
    return load_index(llm_model,vector_store)