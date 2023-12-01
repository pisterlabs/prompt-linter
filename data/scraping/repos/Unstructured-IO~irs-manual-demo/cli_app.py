from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import pinecone
import os


PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")


_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about the Internal Revenue Manuals.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI assistant for answering questions about the Internal Revenue Manuals. You are given the following extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about the war in Internal Revenue Manuals, politely inform them that you are tuned to only answer questions about the Internal Revenue Manuals
Question: {question}
=========
{context}
=========
Answer in Markdown:"""


QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])


def get_chain(vector):
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm)
    return chain


if __name__ == "__main__":
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    vectorstore = Pinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings)
    qa_chain = get_chain(vectorstore)
    chat_history = []
    print("Chat with your docs!")
    while True:
        print("Human:")
        question = input()
        docs = vectorstore.similarity_search(question, k=2)
        result = qa_chain.run(
            input_documents=docs, question=question, chat_history=chat_history
        )
        chat_history.append(result)
        print("AI:")
        print(result)
