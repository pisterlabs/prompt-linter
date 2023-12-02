import openai
openai.api_key = ""

from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.callbacks import CallbackManager

from llamaindex_observer import AimGenericCallbackHandler


# Set up Aim callback
aim_cb = AimGenericCallbackHandler(repo='aim://0.0.0.0:8271')
callback_manager = CallbackManager([aim_cb])

service_context = ServiceContext.from_defaults(callback_manager=callback_manager)

# Create an index
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# Save index to the disk
index.storage_context.persist(persist_dir="index")
