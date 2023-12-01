from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

CHROMA_DB_DIRECTORY = "chroma_db/ask_django_docs"

class Context:
    def __init__(self):
        self.previous_question = None

context = Context()

def answer_query(query):
    embeddings = OpenAIEmbeddings()
    db = Chroma(
        collection_name="ask_django_docs",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIRECTORY
    )

    chat = ChatOpenAI(temperature=0)
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=db.as_retriever(),
        chain_type_kwargs={"verbose": True}
    )
    
    if context.previous_question:
        query = f"{context.previous_question} {query}"
    
    result = chain({"question": query}, return_only_outputs=True)

    context.previous_question = query

    return result

def main():
    while True:
        query = input("Ask a question related to Django (or 'q' to quit): ")
        if query == "q":
            break
        
        result = answer_query(query)

        print(result["answer"])
        print(result["sources"])


if __name__ == "__main__":
    main()