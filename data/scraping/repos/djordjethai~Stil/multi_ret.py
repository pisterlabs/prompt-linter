from langchain.chains.router import MultiRetrievalQAChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.pinecone import Pinecone
import streamlit as st
from Rag_func import hybrid_query, rag, selfquery
import os
import pinecone
import os
import streamlit as st
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from langchain.agents.agent_types import AgentType
from langchain.agents import create_csv_agent
from mojafunkcija import init_cond_llm
from langchain.chains import RetrievalQA


if "rag_index" not in st.session_state:
    st.session_state.rag_index = pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_API_ENV"],
    )
st.session_state.idx = pinecone.Index("embedings1")
# if "hybrid_index" not in st.session_state:
#     st.session_state.hybrid_index = pinecone.init(
#         api_key=os.environ["PINECONE_API_KEY_POS"],
#         environment=os.environ["PINECONE_ENVIRONMENT_POS"],
#     )
#     st.session_state.ind2 = pinecone.Index("bis")

upit = st.text_input("Pitanje: ")


# regular semantic
text = "text"
rag_retriever = Pinecone(
    index=st.session_state.idx,
    embedding=OpenAIEmbeddings(),
    text_key=text,
    namespace="positive",
).as_retriever()

# index = pinecone.Index("bis")
# hybrid search

# hybrid_retriever = PineconeHybridSearchRetriever(
#     index=index,
#     embeddings=OpenAIEmbeddings(),
#     sparse_encoder=BM25Encoder().default(),
#     namespace="pravnikkraciprazan",
#     top_k=3,
#     alpha=0.5,
# )


llm = ChatOpenAI(temperature=0)
# Define metadata fields
metadata_field_info = [
    AttributeInfo(name="title", description="Tema dokumenta", type="string"),
    AttributeInfo(name="keyword", description="reci za pretragu", type="string"),
    AttributeInfo(
        name="text", description="The Content of the document", type="string"
    ),
    AttributeInfo(
        name="source", description="The Source of the document", type="string"
    ),
]

# Define document content description
document_content_description = "Sistematizacija radnih mesta"

index_name = "embedings1"
text = "text"
# Izbor stila i teme
ind3 = pinecone.Index("embedings1")
vector = Pinecone.from_existing_index(
    index_name="embedings1",
    embedding=OpenAIEmbeddings(),
    text_key=text,
    namespace="sistematizacija3",
)
ret = SelfQueryRetriever.from_llm(
    llm,
    vector,
    document_content_description,
    metadata_field_info,
    enable_limit=True,
    verbose=True,
)


retriever_infos = [
    {
        "name": "RAG retriever",
        "description": "Good for answering questions about Positive doo portfolio and services",
        "retriever": rag_retriever,
    },
    {
        "name": "Self Query retriever",
        "description": "Good for answering questions when you mention word: navedi",
        "retriever": ret,
    },
    # {
    #     "name": "Hybrid retriever",
    #     "description": "Good for answering questions when you want to list all ocurrences of a word",
    #     "retriever": hybrid_retriever,
    # },
]


chain = MultiRetrievalQAChain.from_retrievers(
    ChatOpenAI(), retriever_infos, verbose=True
)
odgovor = chain.run(upit)
st.write(odgovor)
