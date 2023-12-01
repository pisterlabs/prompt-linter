import langchain 
langchain.debug = False

import os
from datetime import datetime
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
from streamlit_option_menu import option_menu
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder

from tools.tool_manager import ToolManager
from storage.storage import PersistentStorage
from storage.document import DocumentManager
from storage.logger_config import logger
from ui.sidebar_ui import sidebar
from ui.chat_ui import chat_page
from ui.settings_ui import settings_page
from ui.tools_ui import tools_page
from ui.info_ui import info_page
from ui.settings_ui import list_custom_Agent_names
import config # You can hard code you api keys there
from PIL import Image
import random 




if os.environ["OPENAI_API_KEY"] != '':
    st.session_state.api_keys = True
BASE_DIR= os.path.dirname(os.path.realpath(__file__))
logger.debug('BASE_DIR :'+BASE_DIR)


im = Image.open(BASE_DIR+'/assets/appicon.ico')
st.set_page_config(
    page_title="OmniTool",
    page_icon=im,
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/americium-241/Omnitool_UI/tree/master',
        'Report a bug': "https://github.com/americium-241/Omnitool_UI/tree/master",
        'About': "Prototype for highly interactive and customizable chatbot "
    }
)

#Session_state 

def ensure_session_state():
    logger.debug('Ensure sessions states')
    # Ensure there are defaults for the session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    if "model" not in st.session_state:
        st.session_state.model = "gpt-3.5-turbo"
    if "agent" not in st.session_state:
        st.session_state.agent = AgentType.OPENAI_FUNCTIONS#ZERO_SHOT_REACT_DESCRIPTION
    if 'tool_manager' not in st.session_state:
        st.session_state.tool_manager = ToolManager()
        st.session_state.tool_list = st.session_state.tool_manager.structured_tools
    if "initial_tools" not in st.session_state :
        #Enter a tool title here to make it the initial selected tool, most agents need at least one tool 
        st.session_state.initial_tools=['Testtool']
    if "selected_tools" not in st.session_state : 
        st.session_state.selected_tools = st.session_state.initial_tools
    if "tools" not in st.session_state:
        st.session_state.tools= st.session_state.tool_manager.get_selected_tools(st.session_state.initial_tools)
    if "clicked_cards" not in st.session_state:
        st.session_state.clicked_cards = {tool_name: True for tool_name in st.session_state.initial_tools}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = MessagesPlaceholder(variable_name="chat_history")
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Initialize DocumentManager
    if "doc_manager" not in st.session_state:
        st.session_state.doc_manager = DocumentManager(os.environ["OPENAI_API_KEY"])
    if "documents" not in st.session_state:
        st.session_state.documents = st.session_state.doc_manager.list_documents()
    if "database" not in st.session_state:
        st.session_state.database = st.session_state.doc_manager.database
    if "selected_page" not in st.session_state : 
        st.session_state.selected_page = "Settings"
    if "autorun_state" not in st.session_state:
        st.session_state.autorun_state=False
    if "all_tokens" not in st.session_state:
        st.session_state.all_tokens='' 
    if "prefix" not in st.session_state:
        st.session_state.prefix = ''
    if "suffix" not in st.session_state:
        st.session_state.suffix = ''
    if "session_name" not in st.session_state:
        st.session_state.session_name = {}
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "executed_code" not in st.session_state:
        st.session_state.executed_code=[]
    if "listen" not in st.session_state:
        st.session_state.listen = False
    if "plan_execute" not in st.session_state:
        st.session_state.plan_execute = False
    if "customAgentList" not in st.session_state:
        st.session_state.customAgentList = list_custom_Agent_names


 # menu callback       
def option_menu_cb(cb):
    # For some reason this callback sends a parameter
    st.session_state.selected_page=st.session_state.menu_opt


#@st.cache_resource
def init_storage(db_url='sqlite:///'+BASE_DIR+'//storage//app_session_history.db'):
    logger.info('Building storage and doc_manager')
    # Create or connect to db and initialise document manager
    storage = PersistentStorage(db_url)
    doc_Manager= DocumentManager(os.environ["OPENAI_API_KEY"])
    return storage,doc_Manager
      
# Option Menu
def menusetup():
    
    list_menu=["Chat", "Tools", "Settings","Info"]
    list_pages=[chat_page,tools_page,settings_page,info_page]
    st.session_state.dictpages = dict(zip(list_menu, list_pages))
    list_icons=['cloud','cloud-upload', 'gear','info-circle']
    st.session_state.selected_page = option_menu("",list_menu, 
     icons=list_icons, menu_icon="", orientation="horizontal",
     on_change = option_menu_cb,key='menu_opt',
     default_index=list_menu.index(st.session_state.selected_page))
    
def pageselection(): 
    st.session_state.dictpages[st.session_state.selected_page]()

# Main
def main():
    
    ensure_session_state()
    menusetup()
    st.session_state.storage, st.session_state.doc_manager  = init_storage()
    sidebar()
    pageselection()
  
if __name__ == "__main__":
    main()

