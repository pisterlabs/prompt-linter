import os
import utils
import streamlit as st
from streaming import StreamHandler
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

st.set_page_config(page_title="Chatbot", page_icon="🎥")
st.header('Youtube Chatbot')
st.write('Allows users to chat with youtube videos using the OpenAI LLMs')


class YoutubeChatbot:

    def __init__(self):
        utils.configure_openai_api_key()
        self.openai_model = "gpt-3.5-turbo"

    @st.spinner('Fetching transcript...')
    def setup_qa_chain(self, url):
        # Load documents
        docs = []

        #Fetching Youtube transcripts
        loader = YoutubeLoader.from_youtube_url(url)
        docs.extend(loader.load())

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        # Create embeddings and store in vectordb
        embeddings = OpenAIEmbeddings()
        vectordb = FAISS.from_documents(splits, embeddings)

        # Define retriever
        retriever = vectordb.as_retriever()

        # Setup memory for contextual conversation
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )

        # Setup LLM and QA chain
        llm = ChatOpenAI(model_name=self.openai_model,
                         temperature=0, streaming=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm, retriever=retriever, memory=memory, verbose=True)
        return qa_chain

    @utils.enable_chat_history
    def main(self):
        # User Inputs
        url = st.sidebar.text_input(label='Enter Youtube URLs', value='')

        if not url:
            st.error("Please add links to continue!")
            st.stop()

        user_query = st.chat_input(placeholder="Ask me anything!")

        if url and user_query:
            qa_chain = self.setup_qa_chain(url)

            utils.display_msg(user_query, 'user')
        
            with st.chat_message("assistant"):
                try:
                    st_cb = StreamHandler(st.empty())
                    response = qa_chain.run(user_query, callbacks=[st_cb])
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response})
                except Exception as e:
                     print(e)


if __name__ == "__main__":
    obj = YoutubeChatbot()
    obj.main()
