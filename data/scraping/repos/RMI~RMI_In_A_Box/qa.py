
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import sys


load_dotenv('.env')

persist_directory="./data"
embedding = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding)

# create our Q&A chain
num_docs = 6
pdf_qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'),
    retriever=vectordb.as_retriever(search_kwargs={'k': num_docs}),
    return_source_documents=True,
    verbose=False
)

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"
red = "\033[0;31m"
orange = "\033[0;33;91m"


chat_history = []
print(f"{yellow}---------------------------------------------------------------------------------------------------------------------------")
print('Welcome to v.1.0 of "RMI In A Box". You are now ready to start interacting with the energy transition knowledge base of RMI')
print('---------------------------------------------------------------------------------------------------------------------------')
while True:
    query = input(f"{green}Prompt: ")
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = pdf_qa(
        {"question": query, "chat_history": chat_history})
    print(f"")
    print(f"{white}Answer: " + result["answer"])
    print(f"")
    for i in range(num_docs):
        print(f"{red}From this RMI report: " + result["source_documents"][i].metadata["source"])
        print(f"")
        print(f"{orange}Read more about this on Page #:" + str(result["source_documents"][i].metadata["page"]+1))
        print(f"AND")
   
    chat_history.append((query, result["answer"]))