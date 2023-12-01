from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from get_reviews import GetReviews
import os

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def Ask(apikey, user_question, id):
    os.environ['OPENAI_API_KEY'] = apikey
    url = 'https://www.musinsa.com/app/goods/' + id
    # with open('html.txt') as f:
    #     raw_text = f.read()

    # get review data
    up_reviews, worst_reviews = GetReviews(url, 10)
    raw_text = str(up_reviews) + str(worst_reviews)

    # get the text chunks
    text_chunks = get_text_chunks(raw_text)

    # create vector store
    vectorstore = get_vectorstore(text_chunks)

    # create conversation chain
    conversation_chain = get_conversation_chain(vectorstore)

    response = conversation_chain({"question": user_question})['answer']
    return response
    

# user_question="구매한 사람들의 키/몸무게는 어느 정도야?"
# # user_question="리뷰에서 상품에 대해 유의해야 할 점이 있어?"
# url = 'https://www.musinsa.com/app/goods/3494992'
# Ask(user_question, url)