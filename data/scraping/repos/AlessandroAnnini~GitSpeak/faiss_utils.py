import os
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# from pprint import pprint

chat_history = []


def create_store(repo_name, docs):
    embeddings = OpenAIEmbeddings()

    db_path = f"faiss_index/{repo_name}"

    if os.path.exists(db_path):
        # delete db directory
        print(f"Found existing vector db at {db_path}, deleting...")
        os.system(f"rm -rf {db_path}")

    # create db
    print("Creating vector db...")
    db = FAISS.from_documents(docs, embeddings)

    # save db
    print(f"Saving vector db at {db_path}...")
    db.save_local(db_path)


def get_store(repo_names):
    embeddings = OpenAIEmbeddings()

    db_array = []

    # load all dbs
    for repo_name in repo_names:
        print(f"Loading vector db for {repo_name}...")
        db_path = f"faiss_index/{repo_name}"

        if os.path.exists(db_path):
            print(f"Found existing vector db for {repo_name}, loading...")
            db = FAISS.load_local(db_path, embeddings)
            db_array.append(db)

    if len(db_array) == 0:
        raise ValueError("No db found for the given repo names.")

    # merge all dbs to the first one
    db = db_array[0]
    for i in range(1, len(db_array)):
        db.merge_from(db_array[i])

    return db


def create_chain(db, model="gpt-3.5-turbo"):
    print(f"model: {model}")
    # Create a ChatOpenAI model instance
    llm = ChatOpenAI(model=model, temperature=0.3)

    # Create a retriever from the FAISS instance
    retriever = db.as_retriever(
        search_type="mmr", search_kwargs={"k": 6, "fetch_k": 30}
    )

    # Create a ConversationalRetrievalChain instance
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        return_source_documents=False,
        return_generated_question=False,
    )

    return chain


def search_db(query, chain):
    """Search for a response to the query in the FAISS database."""

    # Query the database
    result = chain({"question": query, "chat_history": chat_history})
    # pprint(result)

    # Add the query and result to the chat history
    chat_history.append((query, result["answer"]))

    # Return the result of the query
    return result["answer"]
