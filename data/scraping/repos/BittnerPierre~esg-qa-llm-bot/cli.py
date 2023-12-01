from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import faiss
import pickle

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    db = pickle.load(f)

db.index = index

chat_history = []
qa = RetrievalQAWithSourcesChain.from_chain_type(llm=ChatOpenAI(temperature=0), retriever=db.as_retriever())

while True:
    # Get user query
    query = input("Enter a query (type 'exit' to quit): ")
    if query.lower() == "exit":
        break

    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    print(f"Answer: {result['answer']}\nSources: {result['sources']}")

print("Exited!!!")