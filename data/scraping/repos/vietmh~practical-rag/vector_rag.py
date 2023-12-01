from langchain.chat_models import ChatOpenAI
from llama_index import (LLMPredictor, ServiceContext, VectorStoreIndex,
                         download_loader)
from llama_index.memory import ChatMemoryBuffer


WikipediaReader = download_loader("WikipediaReader")
wikipedia_reader = WikipediaReader()

documents = wikipedia_reader.load_data(pages=["Cristiano_Ronaldo"])

llm_predictor = LLMPredictor(
    llm=ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
        streaming=True,
    )
)

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    chunk_size=512,
)

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(service_context=service_context)



WikipediaReader = download_loader("WikipediaReader")
wikipedia_reader = WikipediaReader()