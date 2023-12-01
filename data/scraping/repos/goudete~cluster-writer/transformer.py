from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from config.wiki_writer_config import AppConfig

class Transformer():
    
    def __init__(self, splitter):
        self.splitter = splitter

    def transform(self, docs):
        '''
        Split documents into chunks
        Index the chunks (embeddings)
        '''
        texts = self.split(docs)

        # index
        db = Chroma.from_documents(
            texts,
            OpenAIEmbeddings(
                openai_api_key=AppConfig.OPENAI_API_KEY,
                disallowed_special=()
            )
        )
        retriever = db.as_retriever(
            search_type="mmr",  # Also test "similarity"
            search_kwargs={"k": 8},
        )

        # retiever
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=AppConfig.OPENAI_API_KEY)
        qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
        question = "What does this code do?"
        result = qa({"question": question, "chat_history": []})
        print('QUESTION: ', question)
        print('ANSWER: ', result['answer'])
        return

    def split(self, docs):
        texts = self.splitter.split_documents(docs)
        return texts
    