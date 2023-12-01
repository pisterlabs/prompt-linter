import weaviate
from langchain.chains import ChatVectorDBChain
from langchain.llms import OpenAI
from langchain.vectorstores.weaviate import Weaviate

from ai_text_demo.constants import WEAVIATE_URL, OPENAI_API_KEY

client = weaviate.Client(
    url=WEAVIATE_URL,
    additional_headers={
        "X-OpenAI-Api-Key": OPENAI_API_KEY
    }
)

vectorstore = Weaviate(client, "Document", "abstract")

MyOpenAI = OpenAI(temperature=0.2, openai_api_key=OPENAI_API_KEY)

qa = ChatVectorDBChain.from_llm(MyOpenAI, vectorstore)

chat_history = []

while True:
    print("Ask a question: ", end="")
    query = input("")
    print("Answering...")
    result = qa({"question": query, "chat_history": chat_history})
    answer = result["answer"]
    print(f"Answer: {answer}")
    chat_history = [(query, answer)]
