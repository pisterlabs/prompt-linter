import os
from dotenv import load_dotenv

from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# LLamaCpp embeddings from the Alpaca model
from langchain.embeddings import LlamaCppEmbeddings

# FAISS  library for similaarity search
from langchain.vectorstores.faiss import FAISS

import rich
from rich.console import Console
from rich.panel import Panel
from rich import print

########## INITIALIZE RICH CONSOLE  ##################
console = Console()

load_dotenv("../.env")

faiss_file = os.getenv("FAISS_FILE")
gpt4all_path = os.getenv(
    "GPT4ALL_GROOVY_JV13_PATH"
)  # GPT4ALL_GROOVY_JV13_PATH  GPT4ALL_CONVERTED_PATH GPT4ALL_SNOOZY_l13b_PATH
llama_path = os.getenv("LLAMA_PATH")


def similarity_search(query, index):
    # k is the number of similarity searched that matches the query
    # default is 4
    matched_docs = index.similarity_search(query, k=3)
    sources = []
    for doc in matched_docs:
        sources.append(
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    return matched_docs, sources


console.print("[yellow bold] Loading our local index vector db")
# create the embedding object
embeddings = LlamaCppEmbeddings(model_path=llama_path)
# Load our local index vector db
index = FAISS.load_local(faiss_file, embeddings)

console.print("[yellow bold] Loading GPT4ALL model")
# create the GPT4All llm object
callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(
    model=gpt4all_path, callbacks=callbacks, backend="gptj", verbose=True
)  # backend="gptj",


# create the prompt template
# template = """
# Please use the following context to answer questions.
# Context: {context}
# ---
# Question: {question}
# Answer: Let's think step by step."""

# Hardcoded question
question = "Why Milvus Vector Database is used in corporate brain?"

# matched_docs, sources = similarity_search(question, index)
# Creating the context
# context = "\n".join([doc.page_content for doc in matched_docs])
# print(f"Context: {context}")
# print("Instantiating the prompt template and the GPT4All chain")
# prompt = PromptTemplate(
#     template=template, input_variables=["context", "question"]
# ).partial(context=context)
# prompt = PromptTemplate(template=template, input_variables=["context", "question"])
# llm_chain = LLMChain(prompt=prompt, llm=llm)
# Print the result
# print(llm_chain.run(question))

console.print("[yellow bold] Creating chain")
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.as_retriever(),
    input_key="question",
)

console.print("[yellow bold] Running Chain")
responses = chain.run(question)
print(responses)
