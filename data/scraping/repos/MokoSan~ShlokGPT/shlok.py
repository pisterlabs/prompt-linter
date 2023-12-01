from typing import Dict
import streamlit as st
import uuid
import os
from streamlit_chat import message
import requests
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv
from supabase import create_client, Client

class ShlokGPT(object):
    def __init__(self):
        # Create the Supabase Client.
        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_KEY")
        self.supabase: Client = create_client(url, key)

        # Create the ChatOpenAI Chain.
        openai_key = os.getenv("OPENAI_API_KEY") 
        self.llm = ChatOpenAI(openai_api_key=openai_key, temperature=0, model='gpt-3.5-turbo')
        self.human_template = HumanMessagePromptTemplate.from_template(
        '''Generate a 6 line sanskrit slok about the following topics: {topics}. 

        The output should consist of the shlok followed by the translation in the following format:

        Shlok
        Translation:
        ''')
        self.chat_prompt = ChatPromptTemplate.from_messages([self.human_template])

    def chat(self, topics: str) -> Dict[str, str]:
        topics = topics.strip()
        chat_prompt_value = self.chat_prompt.format_prompt(topics = topics.strip())
        messages = chat_prompt_value.to_messages()
        result = self.llm(messages)
        content = result.content
        split = content.split('Translation:')
        shlok = split[0]
        translation = split[1]
        return {'shlok': shlok, 'translation' : translation}

    def generate_audio(self, text) -> str:
        try:
            options = { 
                'headers': { 'Accept': 'application/octet-stream', 'Content-Type': 'text/plain', 'x-api-key': os.getenv('NARKEET_API_KEY'), },
                'data': text.encode('utf8')
            }
            url = f'https://api.narakeet.com/text-to-speech/m4a?voice=amitabh'
            file_data = requests.post(url, **options).content
            file_name = str(uuid.uuid4()) + ".m4a"
            res = self.supabase.storage.from_('shloks').upload(file_name, file_data)
            url = self.supabase.storage.from_('shloks').get_public_url(file_name)
            return url 
        except Exception as ex:
            print(ex)

st.set_page_config( page_title="Shlok-GPT: Generate Sanskrit Shloks with AI.",)
st.title("Shlok-GPT: Generate Sanskrit Shloks with AI.")

def process_input():

    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) <= 0:
        return

    user_text = st.session_state["user_input"].strip()
    with st.session_state["thinking_spinner"], st.spinner(f"Generating Shloks for: {user_text}."):
        try:
            st.session_state["messages"].append((user_text, True))
            shlok_gpt = st.session_state["ShlokGPT"]
            result = shlok_gpt.chat(user_text)
            content = result['shlok'] + " " + result['translation']
            st.session_state["messages"].append((content, False))
            file = shlok_gpt.generate_audio(result['shlok'])
            st.session_state["messages"].append((f'<audio controls src="{file}"></audio>', False))

        except Exception as e:
            print(e)
            st.session_state["messages"].append((f"Failed for: {user_text}. Full exception: {e} ", False))

def display_messages() -> None:
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        avatar_style = 'initials/svg?seed=OM'
        if is_user:
            avatar_style = 'initials/svg?seed=User' 
        message(msg, is_user=is_user, key = f"{i}.{msg}", allow_html = True, avatar_style = avatar_style) 

    st.session_state["thinking_spinner"] = st.empty() 

def run():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") 
        st.session_state["ShlokGPT"] = ShlokGPT()

    display_messages()
    st.text_input("Enter topics to generate Shloks for:", key="user_input", on_change=process_input, label_visibility='visible')

    st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True )
    st.divider()
    st.text("Made by Moko - https://twitter.com/MokoSharma.")

if __name__ == '__main__':
    load_dotenv()
    run()