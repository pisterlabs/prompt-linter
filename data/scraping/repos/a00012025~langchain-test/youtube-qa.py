from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from llama_index import download_loader
# import os
# os.environ["OPENAI_API_KEY"] = ''

file_path = input('Enter the youtube link: ')

YoutubeTranscriptReader = download_loader("YoutubeTranscriptReader")
documents = YoutubeTranscriptReader().load_data(ytlinks=[file_path])

documents = [doc.to_langchain_format() for doc in documents]
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(client=None)
retriever = FAISS.from_documents(documents, embeddings).as_retriever(k=4)

llm=ChatOpenAI(temperature=1, model="gpt-3.5-turbo", max_tokens=2048, client=None)

memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    return_messages=True,
    k=6
)

conversation = ConversationalRetrievalChain.from_llm(
    llm=llm, 
    retriever=retriever,
    # verbose=True, 
    memory=memory,
    max_tokens_limit=1536  
)

def chatbot(pt):
    res = conversation({'question': pt})['answer']
    return res

if __name__=='__main__':
    while True:
        print('########################################\n')
        pt = input('ASK: ')
        if pt.lower()=='end':
            break
        response = chatbot(pt)
        print('\n----------------------------------------\n')
        print('ChatGPT says: \n')
        print(response, '\n')