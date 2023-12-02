import os

os.environ["SERPAPI_API_KEY"] = '...'
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

ENV_HOST = "https://cloud.langfuse.com"
ENV_SECRET_KEY = "sk-lf-..."
ENV_PUBLIC_KEY = "pk-lf-..."
handler = CallbackHandler(ENV_PUBLIC_KEY, ENV_SECRET_KEY, ENV_HOST)
urls = [
    "https://raw.githubusercontent.com/langfuse/langfuse-docs/main/public/state_of_the_union.txt",
]

loader = UnstructuredURLLoader(urls=urls)

llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
docsearch = Chroma.from_documents(texts, embeddings)

query = "What did the president say about Ketanji Brown Jackson"

chain = RetrievalQA.from_chain_type(
    llm,
    retriever=docsearch.as_retriever(search_kwargs={"k": 1}),
)

result = chain.run(query, callbacks=[handler])

print(result)

handler.langfuse.flush()


from langchain.agents import AgentType, initialize_agent, load_tools


handler = CallbackHandler(ENV_PUBLIC_KEY, ENV_SECRET_KEY, ENV_HOST)
llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
result = agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?", callbacks=[handler])
handler.langfuse.flush()
print("output variable: ", result)
