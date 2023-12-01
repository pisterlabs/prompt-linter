from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from langchain import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
import os
from HC_bot import HC_bot


load_dotenv()



def get_vectorstore(embedding_model, db_name):
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
    db = Chroma(persist_directory=db_name, embedding_function=embeddings)
    return db


def set_custom_prompt(history, context, question):
    """
    Prompt template for QA retrieval for each vectorstore
    """


    custom_prompt_template = f"""Use the context to answer the user's question. Use the history to know what has been discussed.
    If the context can't answer the question, say that you don't know.
    Chat history: {history}
    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:"""

    return custom_prompt_template



def get_matching_docs(question):
    db = get_vectorstore("sentence-transformers/all-MiniLM-L6-v2", "chroma_db")
    matching_docs = db.similarity_search(question)
    return matching_docs


def create_chatbot():
    email = os.getenv("EMAIL_HF")
    pw = os.getenv("PASS_HF")
    bot = HC_bot(email, pw)
    bot.new_chat()

    return bot


def ask_bot(question, history):
    matching_docs = get_matching_docs(question)
    context = matching_docs[0].page_content

    prompt = set_custom_prompt(history, context, question)

    bot = create_chatbot()
    ans = bot.one_chat(prompt)

    return ans


def chat():
    history = ""
    query = input(">>> ")

    while query != "quit":
        ans = ask_bot(query, history)
        add_to_hist = query + ans
        history = history + add_to_hist

        print("=====================================================")
        print(ans)
        print("=====================================================")

        query = input(">>> ")









def main():

    chat()



    """
    qa_result = qa_bot()
    while True:
        query = input("Question: ")
        if query == "quit":
            break

        response = qa_result({'query': query})

        print(response)
        """








if __name__=='__main__':
    main()

