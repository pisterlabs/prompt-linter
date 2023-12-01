from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext, LangchainEmbedding
from langchain.embeddings import CohereEmbeddings

from llama_index.llms import Replicate
llama2 = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
llm = Replicate(
    model=llama2,
    temperature=0.01,
    additional_kwargs={"top_p": 1, "max_new_tokens": 300}
)

embed_model = LangchainEmbedding(CohereEmbeddings())
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=embed_model)

documents = SimpleDirectoryReader("./data").load_data()
local_index = VectorStoreIndex.from_documents(
    documents, service_context=service_context)

local_index.storage_context.persist()