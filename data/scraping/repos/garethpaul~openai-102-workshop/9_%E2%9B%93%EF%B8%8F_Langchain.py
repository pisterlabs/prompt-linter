import streamlit as st

st.markdown("# Langchain")

st.markdown("# What is langchain?")
st.markdown("""
Library that can “chain” together different components to create more advanced use cases around LLMs:

*Prompt templates*
Prompt templates are templates for different types of prompts. Like “chatbot” style templates, ELI5 question-answering, etc

*LLMs*
Large language models like GPT-3, BLOOM, etc

*Agents*
Agents use LLMs to decide what actions should be taken. Tools like web search or calculators and all packaged into a logical loop of operations

*Memory*
Short-term memory, long-term memory

And More!

""")

st.markdown("## Examples")
st.markdown("""### Install Deps""")
st.code("""
!pip install langchain
!pip install getpass4
!pip install tiktoken
!pip install openai
""")

st.markdown("### Set Environment Variables")
st.code("""
import os
from getpass import getpass

os.environ["OPENAI_API_KEY"] = getpass('Enter the secret value: ')
""")

st.markdown("""### Writing a prompt """)
st.code("""
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.9) 
text = "What would be a good company name for a company that makes colorful socks?"
print(llm(text))
""")

st.markdown("""### Writing a prompt with a template""")
st.code("""
from langchain.prompts import PromptTemplate

template = "Question: {question}

Let's think step by step.

Answer:"

prompt = PromptTemplate(template=template, input_variables=["question"])
""")

st.markdown("### Agents")
st.markdown("""
Agents involve an LLM making decisions about which Actions to take, taking that Action, seeing an Observation, and repeating that until done.

When used correctly agents can be extremely powerful. In order to load agents, you should understand the following concepts:

- Tool: A function that performs a specific duty. This can be things like: Google Search, Database lookup, Python REPL, other chains.
- LLM: The language model powering the agent.
- Agent: The agent to use.

Tools: https://python.langchain.com/en/latest/modules/agents/tools.html
Agent Types: https://python.langchain.com/en/latest/modules/agents/agents/agent_types.html
""")


st.code("""
from langchain.agents import load_tools
from langchain.agents import initialize_agent
""")

st.code("""!pip install wikipedia""")

st.code("""
llm = OpenAI(temperature=0)
tools = load_tools(["wikipedia", "llm-math"], llm=llm)
""")

st.code("""agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)""")
st.code("""agent.run("In what year was the film Departed with Leopnardo Dicaprio released? What is this year raised to the 0.43 power?")""")

st.markdown("""### Memory""")
st.code("""
from langchain import OpenAI, ConversationChain

llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)

conversation.predict(input="Hi there!")
""")

st.code("""
conversation.predict(input="Can we talk about AI?")
"""
        )

st.markdown("# Document Loaders")
st.markdown("""
Combining language models with your own text data is a powerful way to differentiate them. The first step in doing this is to load the data into “documents” - a fancy way of say some pieces of text. This module is aimed at making this easy.

https://python.langchain.com/en/latest/modules/indexes/document_loaders.html
""")

st.code("""
from langchain.document_loaders import NotionDirectoryLoader

loader = NotionDirectoryLoader("Notion_DB")

docs = loader.load()
""")

st.markdown("# Document Indexes")
st.markdown("""
Indexes refer to ways to structure documents so that LLMs can best interact with them. This module contains utility functions for working with documents

*Embeddings*: An embedding is a numerical representation of a piece of information, for example, text, documents, images, audio, etc.
*Text Splitters*: When you want to deal with long pieces of text, it is necessary to split up that text into chunks.
*Vectorstores*: Vector databases store and index vector embeddings from NLP models to understand the meaning and context of strings of text, sentences, and whole documents for more accurate and relevant search results.
""")

st.code("""
import requests

url = "https://raw.githubusercontent.com/hwchase17/langchain/master/docs/modules/state_of_the_union.txt"
res = requests.get(url)
with open("state_of_the_union.txt", "w") as f:
  f.write(res.text)
""")

st.code("""
# Text Splitter
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
""")

st.code("""
!pip install sentence_transformers
""")

st.code("""
# Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

text = "This is a test query."
query_result = embeddings.embed_query(text)

print(query_result)
print(len(query_result))
""")

st.code("""
!pip install faiss-cpu
""")

st.code("""
from langchain.vectorstores import FAISS

db = FAISS.from_documents(docs, embeddings)

query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
""")

st.code("""
print("Docs returned: ",len(docs))
print("\n==============\n")
print(docs[0].page_content)
print("\n==============\n")
print(docs[0].metadata)
""")

st.code("""
db.save_local("faiss_index")
new_db = FAISS.load_local("faiss_index", embeddings)
docs = new_db.similarity_search(query)
print(docs[0].page_content)
""")
