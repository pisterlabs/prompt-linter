import anthropic
import streamlit as st
from streamlit_chat import message
from streamlit.logger import get_logger
from typing import Any, Dict, List
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.vectorstores import SupabaseVectorStore
from llm import LANGUAGE_PROMPT
from stats import add_usage


class AnswerConversationBufferMemory(ConversationBufferMemory):
    """ref https://github.com/hwchase17/langchain/issues/5630#issuecomment-1574222564"""
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        return super(AnswerConversationBufferMemory, self).save_context(
            inputs, {'response': outputs['answer']})


memory = AnswerConversationBufferMemory(
            memory_key="chat_history", return_messages=True)
openai_api_key = st.secrets.openai_api_key
logger = get_logger(__name__)


def count_tokens(question, model):
    count = f'Words: {len(question.split())}'
    return count


def chat_with_doc(model, vector_store: SupabaseVectorStore, stats_db):
    
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
        
    question = st.text_input("## いかがいたしましたか？")
    columns = st.columns(3)
    with columns[0]:
        button = st.button("決定")
    with columns[1]:
        count_button = st.button("トークンを数える", type='secondary')
    with columns[2]:
        clear_history = st.button("チャット履歴を消す", type='secondary')
    
    if clear_history:
        # Clear memory in Langchain
        memory.clear()
        st.session_state['chat_history'] = []
        st.experimental_rerun()

    if button:
        qa = None
        add_usage(stats_db, "chat", question, {"model": model, "temperature": st.session_state['temperature']})
        ConversationalRetrievalChain.prompts = LANGUAGE_PROMPT
        
        logger.info('Using OpenAI model %s', model)
        qa = ConversationalRetrievalChain.from_llm(
            OpenAI(
                model_name=st.session_state['model'], openai_api_key=openai_api_key, temperature=st.session_state['temperature'], max_tokens=st.session_state['max_tokens']), vector_store.as_retriever(), memory=memory, verbose=True, return_source_documents=True)
        
        st.session_state['chat_history'].append(("You", question))

        # Generate model's response and add it to chat history
        model_response = qa({"question": question})
        logger.info('Result: %s', model_response)

        st.session_state['chat_history'].append(("Akasha", model_response["answer"], ))

        # Display chat history
        st.empty()
        is_user = True
        for speaker, text in st.session_state['chat_history']:
            # st.markdown(f"**{speaker}:** {text}")
            if speaker == "You":
                is_user = True
            else:
                is_user = False
            message(text, is_user=is_user)
        st.markdown("""
                    ---
                    Source:
                    """)
        st.write(model_response["source_documents"])
        
    if count_button:
        st.write(count_tokens(question, model))