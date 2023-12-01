"""
For running general code with Llama llm
"""

import chromadb
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores.chroma import Chroma
from datasets import load_dataset
from chromadb.config import Settings
from langchain.prompts import ChatPromptTemplate
from langchain.llms.llamacpp import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts.prompt import Prompt
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import time
import pathlib
import os
import guidance
from guidance import gen, select, Guidance
from guidance.models._llama_cpp import LlamaCpp as gLlamaCpp
from llama_cpp import Llama, ChatCompletionMessage
from argparse import ArgumentParser
from modules.ai.utils.vectorstore import create_vectorstore
from modules.ai.utils.llm import create_llm
import json

# from langchain.vectorstores.chroma import Chroma
def run_llm():
    vectorstore = create_vectorstore()
    llm = create_llm()

# You will be talking to a student taking the course D7032E. Use the following pieces of retrieved context to answer the question. 

# Course summary: The course will have an emphasis on selected topics from: Project planning and management, problem analysis,
# software management and interpretation, code complexity, API design, debugging and testing, configuration
# management, documentation, design patterns, build support and tools of the trade, packaging, release management
# and deployment, modeling and structuring of software, reuse, components, architectures, maintenance and
# documentation. The course includes a number of assignments, which are to be completed in groups, and that are
# evaluated in both written and oral form. Individual examination is given through tests and a home exam. 

# You will be talking to a student taking the AI course D0038E. Use the following pieces of retrieved context to answer the question. 
    prompt_str = """Human: You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise. 
Don't directly refer to the context text, pretend like you already knew the context information.

Question: {question}

Context: {context}

Answer:"""

    questions: list[str] = [
        # "In lab 6 do we use boosting? ",
        # "Explain what we are doing in lab 6 task 1.",
        # "In lab 6 task 1 what is the expected difference in performance between the two models?",
        # "For lab 6 summarize task 6.",
        # "What models are used in lab 6?",
        # "For task 7 in in lab 6 give some examples of models i can experiment on.",
        # "Are we allowed to do lab 6 outside the lab sessions?",
        # "In lab 6, in what website can i read more about the different models?",
        # "What program are we supposed to use for lab 6?",
        # "in lab 6 what is task 4?",

        # "In Lab3 what is the excercise about?",
        # "What kind of classifier will Lab3 be about?",
        # "What operator can be used in rapidminer to take data and a pretrained model and get labeled dataset as an output?",
        # "Give me an example of a hyperparameter",
        # "What is a k-nearest neighbors classifier?",
        # "How many tasks are there in lab3?",
        # "What dataset do you need to load for task 4?",
        # "How does the K-NN model work?",
        # "What happens when the dimensions increase when using k-NN?",
        # "Are there any extra tasks in lab3?",
        # "Summarize lab 6.",
        "What is SOLID principles?"
    ]

    # llm("Finish the sentence: I compare thee [...]")
        # "In lab 6 do we use boosting? ",
        # "Explain what we are doing in lab 6 task 1.",
        # "In lab 6 task 1 what is the expected difference in performance between the two models?",
        # "For lab 6 summarize task 6.",
        # "What models are used in lab 6?",
        # "For task 7 in in lab 6 give some examples of models i can experiment on.",
        # "Are we allowed to do lab 6 outside the lab sessions?",
        # "In lab 6, in what website can i read more about the different models?",
        # "What program are we supposed to use for lab 6?",
        # "in lab 6 what is task 4?",

        # "In Lab3 what is the excercise about?",
        # "What kind of classifier will Lab3 be about?",
        # "What operator can be used in rapidminer to take data and a pretrained model and get labeled dataset as an output?",
        # "Give me an example of a hyperparameter",
        # "What is a k-nearest neighbors classifier?",
        # "How many tasks are there in lab3?",
        # "What dataset do you need to load for task 4?",
        # "How does the K-NN model work?",
        # "What happens when the dimensions increase when using k-NN?",
        # "Are there any extra tasks in lab3?",
        # "Summarize lab 6.",
    #     "What is SOLID principles?"
    # ]

    # llm("Finish the sentence: I compare thee [...]")

    for question in questions:
        docs = vectorstore.similarity_search(question, k=2,filter={'course':'D7032E'})
        context = ""
        print(f"Docs", docs)
        print(f"Docs: {len(docs)}")
        for doc in docs:
            print('doc')
            # print("Doc id:", (doc.metadata["id"],doc.metadata["chunk-id"]))
            print("Doc metadata:", doc.metadata)
            context += doc.page_content
        resulting_prompt = prompt_str.format(question = question, context = context)
        # resulting_prompt = prompt_no_context_str.format(question = question)
        print("Full prompt (length: {length}):".format(length=len(resulting_prompt)))
        print(resulting_prompt+"\n")
        print(f"############## Start")
        print(f"Question: {question}\n")

        print(f"Answer: ",end="")
        llm(resulting_prompt+"\n")
        print("\n")
        print(f"############## Finished\n\n")




if __name__ == "__main__":
    run_llm()
    