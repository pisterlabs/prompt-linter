import os
# from dotenv import load_dotenv
import streamlit as st
import tiktoken

# from streamlit_chat import message
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


data_dir = "data/" 
openai_api_key = os.environ.get('OPENAI_API_KEY')


@st.cache_resource(show_spinner=False)
def load_vectorstore(data_dir):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True})  
        
    vectorstore = FAISS.load_local(data_dir, embeddings)
    return vectorstore


def get_conversation_chain(vetorestore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo-1106', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h[:50],
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain


def main():
    st.set_page_config(page_title="HistoryChat", page_icon=":books:")
    st.title("_History :red[QA Chat]_ :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
        
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "역사에 관해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None    
        st.session_state.vectorstore = load_vectorstore("faiss_index")
        st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore, openai_api_key) 
        st.session_state.processComplete = True
        

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                response = result['answer']
                st.markdown(response)

                
        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


# 메인 함수 호출
if __name__ == "__main__":
    main()