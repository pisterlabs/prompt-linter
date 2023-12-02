from fastapi import HTTPException, status
from typing import List
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from sqlalchemy.orm import Session

from src.conf import messages
from src.database.models import User
from src.repository.chats import get_chat_by_id
from dotenv import load_dotenv

load_dotenv()


async def get_context(chat_id: int, db: Session, user: User) -> List:

    chat = await get_chat_by_id(chat_id, db, user)
    context = chat.file_url
    if context is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=messages.NOT_FOUND)

    with open(context, "r", encoding="utf-8") as file:
        text = file.read()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    return chunks


def get_conversation_chain(user_question, context) -> str:

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=context, embedding=embeddings)

    llm = ChatOpenAI(model='gpt-3.5-turbo')
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )

    response = conversation_chain({'question': user_question})
    chat_history = response['chat_history']

    # for separate question/answer items
    response_dict = {'user_messages': [], 'bot_messages': []}

    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            response_dict['user_messages'].append(message.content)
        else:
            response_dict['bot_messages'].append(message.content)

    return str(response_dict)
