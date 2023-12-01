# %% [markdown]
# # LangChain: Q&A over Documents
# 
# An example might be a tool that would allow you to query a product catalog for items of interest.

# %%
#pip install --upgrade langchain

# %%
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv('../.env')) # read local .env file

# %%
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown

# %%
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)

# %%
from langchain.indexes import VectorstoreIndexCreator

# %%
#pip install docarray

# %%
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

# %%
query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."

# %%
response = index.query(query)

# %%
display(Markdown(response))

# %%
loader = CSVLoader(file_path=file)

# %%
docs = loader.load()

# %%
docs[0]

# %%
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# %%
embed = embeddings.embed_query("Hi my name is Harrison")

# %%
print(len(embed))

# %%
print(embed[:5])

# %%
db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)

# %%
query = "Please suggest a shirt with sunblocking"

# %%
docs = db.similarity_search(query)

# %%
len(docs)

# %%
docs[0]

# %%
retriever = db.as_retriever()

# %%
llm = ChatOpenAI(temperature = 0.0)


# %%
qdocs = "".join([docs[i].page_content for i in range(len(docs))])


# %%
response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.") 


# %%
display(Markdown(response))

# %%
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)

# %%
query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."

# %%
response = qa_stuff.run(query)

# %%
display(Markdown(response))

# %%
response = index.query(query, llm=llm)

# %%
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([loader])
