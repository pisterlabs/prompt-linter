from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.chat_engine.types import ChatMode
from llama_index.llms import OpenAI, Anthropic

service_context = ServiceContext.from_defaults(llm=OpenAI())
data = SimpleDirectoryReader(input_dir="../data/paul_graham/").load_data()

index = VectorStoreIndex.from_documents(data, service_context=service_context)

chat_engine = index.as_chat_engine(
    service_context=service_context, chat_mode=ChatMode.REACT, verbose=True
)

response = chat_engine.chat(
    "Use the tool to answer what did Paul Graham do in the summer of 1995?"
)
print(response)