from langchain.llms import Ollama
from langchain.embeddings import GPT4AllEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_texts(
    ["Bread - (0.0, 1.0, 2.0)\nMilk - (1.0, 2.0, 0.0)"], embedding=GPT4AllEmbeddings()
)
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = Ollama(model="mistral")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
)

for s in chain.stream("Where is the bread?"):
    print(s, end="", flush=True)
