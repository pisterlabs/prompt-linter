import streamlit as st 
import os

from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import TextLoader
import autogen

from tools.utils import get_class_func_from_module,monitorFolder
from config import KEYS,MODELS,AGENTS

Local_DIR= os.path.dirname(os.path.realpath(__file__))

monitored_files=monitorFolder(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..', 'agents','agents_list'))

list_custom_Agent=[]

for f in monitored_files:
    class_agent,_=get_class_func_from_module(f)
    list_custom_Agent.extend(class_agent)

list_custom_Agent_names=[a[0] for a in list_custom_Agent]
AGENTS.extend(list_custom_Agent_names)


def api_keys_init():
    with st.expander(label='API keys',expanded = True):
        
        if "api_keys" not in st.session_state :
            for i,k in enumerate(KEYS): 
                os.environ[k] = st.text_input(k, type="password")
            st.session_state.api_keys = True
            
            st.info("Please add your OpenAI API key to continue.")        
            st.stop()
        else : 
            for k in KEYS: 
                os.environ[k] = st.text_input(k, type="password",value=os.environ[k])

#Necessary to have callbacks to keep states updated
def llm_menu_cb():
    st.session_state.model=st.session_state.menu_model
def agent_menu_cb():
    st.session_state.agent=st.session_state.menu_agent
def prefixprompt_menu_cb():
    st.session_state.prefix=st.session_state.menu_prefixprompt
def suffixprompt_menu_cb():
    st.session_state.suffix=st.session_state.menu_suffixprompt

def listen_menu_cb():
       st.session_state.listen=st.session_state.menu_listen

def planexec_menu_cb():
       st.session_state.plan_execute=st.session_state.menu_plan_exec

def empty_vdb():
        keys=list(st.session_state.doc_manager.documents.keys())
        for i,doc in enumerate(keys):
            st.session_state.doc_manager.remove_document(doc)
            st.session_state.doc_manager.database.delete([st.session_state.doc_manager.database.index_to_docstore_id[i]])
            
# Subpage functions 
def file_upload():

    with st.expander(label='Load documents',expanded = False):
        col1,col2=st.columns([100,92])
        
        uploaded_files = col1.file_uploader("Embbedings Files", type=['txt','pdf'], accept_multiple_files=True)
        uploaded_files_workspace = col2.file_uploader("Workspace Files", type=['txt','pdf','csv','png','jpg'], accept_multiple_files=True)
        
        if uploaded_files:
            load_files(uploaded_files)
        if uploaded_files_workspace:
            load_files_workspace(uploaded_files_workspace)
        
        col1.button('Empty document db',on_click=empty_vdb)
        if len(list(st.session_state.doc_manager.documents.keys()))>0 : 
            col1.write('Documents loaded : \n'+str(list(st.session_state.doc_manager.documents.keys()))[1:-1])
     
@st.cache_data()
def load_files_workspace(uploaded_files):
    workspace_dir=Local_DIR+'\\..\\workspace\\'
    for file in uploaded_files: 
        load_file_workspace(file,workspace_dir)

@st.cache_data()
def load_file_workspace(file,workspace_dir):
        file_value=file.getvalue()
        with open(workspace_dir+str(file.name), 'wb') as f:
            f.write(file_value)
        


@st.cache_data()
def load_files(uploaded_files):

    for uploaded_file in uploaded_files:
        load_file(uploaded_file)
    
    st.session_state.doc_manager.create_embeddings_and_database()
    st.session_state.documents = st.session_state.doc_manager.list_documents()
    st.session_state.database = st.session_state.doc_manager.database
    st.success("Documents loaded and embeddings created.")

@st.cache_data()
def load_file(uploaded_file):
    if uploaded_file.type == 'application/pdf':
        temp_file_path = f"./temp_{uploaded_file.name}"
        with open(temp_file_path, mode='wb') as w:
                w.write(uploaded_file.getvalue())
        loader = PyMuPDFLoader(temp_file_path)
        doc_content = loader.load()
        st.session_state.doc_manager.add_document(uploaded_file.name, doc_content)
        os.remove(temp_file_path)  # remove temporary file after loading

    if uploaded_file.type == 'text/plain' : 
    # Write uploaded file to a temporary file and load with TextLoader
        temp_file_path = f"./temp_{uploaded_file.name}"
        with open(temp_file_path, 'w') as f:
            f.write(uploaded_file.read().decode('utf-8'))
        loader = TextLoader(temp_file_path)
        doc_content = loader.load()
        st.session_state.doc_manager.add_document(uploaded_file.name, doc_content)
        os.remove(temp_file_path)  # remove temporary file after loading
            


def settings_page():
   
    api_keys_init()
    
    with st.expander(label='Settings',expanded = True):
        col1,col2=st.columns(2)
        st.session_state.agent = col1.selectbox("Select agent", options=AGENTS,key='menu_agent',on_change=agent_menu_cb,index=AGENTS.index(st.session_state.agent),help=Agent_Description[str(st.session_state.agent)])
        st.session_state.model = col2.selectbox("Select a model", options=MODELS,key='menu_model',on_change=llm_menu_cb ,index=MODELS.index(st.session_state.model ))  
        st.session_state.prefix=col1.text_area('Prefix',key='menu_prefixprompt',on_change=prefixprompt_menu_cb,value= st.session_state.prefix,placeholder='First input for initial prompt')
        st.session_state.suffix=col2.text_area('Suffix',key='menu_suffixprompt',on_change=suffixprompt_menu_cb,value= st.session_state.suffix,placeholder='Last input for initial prompt')
    file_upload()
    with st.expander(label='Experimental',expanded = False):
        col1,col2,col3=st.columns(3)
        st.session_state.listen= col1.checkbox('Start Listening',key='menu_listen',on_change=listen_menu_cb,value=st.session_state.listen )
        st.session_state.plan_execute= col2.checkbox('Plan and execute',key='menu_plan_exec',on_change=planexec_menu_cb,value=st.session_state.plan_execute )
    make_autogen_config()

def make_autogen_config():
    param="""        "model": "{model}",
        "api_key": "{key}" """.format(model=st.session_state.model,key=os.environ["OPENAI_API_KEY"])
    # Make autogen llm_config
    json_string = """[
        {\n"""+str(param)+"""
              }
    ]"""

    # Store the JSON string in an environment variable
    os.environ['OAI_CONFIG_LIST'] = json_string
    config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")
    llm_config={
            #"seed": 42,  # seed for caching and reproducibility
            "config_list": config_list,  # a list of OpenAI API configurations
            "temperature": 0,  # temperature for sampling
        }
    st.session_state.autogen_llm_config=llm_config



Agent_Description={
'AgentType.OPENAI_FUNCTIONS':
"Certain OpenAI models (like gpt-3.5-turbo-0613 and gpt-4-0613) have been explicitly fine-tuned to detect when a function should be called and respond with the inputs that should be passed to the function. The OpenAI Functions Agent is designed to work with these models.",

'AgentType.ZERO_SHOT_REACT_DESCRIPTION':
"This agent uses the ReAct framework to determine which tool to use based solely on the tool's description. Any number of tools can be provided. This agent requires that a description is provided for each tool.",
'AgentType.CONVERSATIONAL_REACT_DESCRIPTION':
"This agent is designed to be used in conversational settings. The prompt is designed to make the agent helpful and conversational. It uses the ReAct framework to decide which tool to use, and uses memory to remember the previous conversation interactions.",
'AgentType.SELF_ASK_WITH_SEARCH':
"This agent utilizes a single tool that should be named Intermediate Answer. This tool should be able to lookup factual answers to questions. This agent is equivalent to the original self ask with search paper, where a Google search API was provided as the tool.",
'AgentType.REACT_DOCSTORE':
"This agent uses the ReAct framework to interact with a docstore. Two tools must be provided: a Search tool and a Lookup tool (they must be named exactly as so). The Search tool should search for a document, while the Lookup tool should lookup a term in the most recently found document. This agent is equivalent to the original ReAct paper, specifically the Wikipedia example.",
'AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION':
"The structured tool chat agent is capable of using multi-input tools.",
}

for ag in AGENTS:
    if str(ag) not in Agent_Description.keys():
         Agent_Description.update({str(ag):'No description'})
