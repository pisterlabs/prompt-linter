import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from traceloop.sdk import Traceloop

Traceloop.init(disable_batch=True)

index_name = 'gpt-4-langchain-docs-fast'
model_name = 'text-embedding-ada-002'

index = pinecone.Index(index_name)

embed = OpenAIEmbeddings(
    model=model_name,
)

vectorstore = Pinecone(
    index, embed.embed_query, "text"
)

llm = ChatOpenAI(
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

print(qa.run("how do I build an agent with LangChain?"))
