import time
from dive.util.configAPIKey import set_hugging_face_auth,set_pinecone_api_key,set_pinecone_env,set_pinecone_index_dimentions
from huggingface_hub import hf_hub_download
import os
from langchain.callbacks.manager import CallbackManager
from examples.base import index_example_data,query_example_data,clear_example_data
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import LlamaCpp

#Use chromadb and Llama 2 embeddings and llm

set_hugging_face_auth()
hf_auth = os.environ.get('use_auth_token', '')
model_path = hf_hub_download(repo_id='TheBloke/Llama-2-7B-GGML', filename='llama-2-7b.ggmlv3.q5_1.bin', use_auth_token=hf_auth)

set_pinecone_api_key()
set_pinecone_env()
set_pinecone_index_dimentions()

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llama_embeddings = LlamaCppEmbeddings(model_path=model_path)
llm = LlamaCpp(
    model_path=model_path,
    input={"temperature": 0, "max_length": 2000, "top_p": 1},
    callback_manager=callback_manager,
    verbose=True,
)
index_example_data(256, 20, False, llama_embeddings,llm)
print('------------Finish Indexing Data-----------------')
time.sleep(30)
print('------------Start Querying Data-----------------')
question='What did the author do growing up?'
instruction=None #'summarise your response in no more than 5 lines'
query_example_data(question,4, llama_embeddings,llm,instruction)
#clear_example_data()