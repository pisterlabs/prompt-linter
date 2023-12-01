from llama_index import LLMPredictor, GPTVectorStoreIndex, SimpleDirectoryReader, ServiceContext
from langchain.chat_models import ChatOpenAI

llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=1024))

#load docs
documents = SimpleDirectoryReader('local-data').load_data()

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

index.storage_context.persist(persist_dir="local-index")
print("Saved embeddings")
