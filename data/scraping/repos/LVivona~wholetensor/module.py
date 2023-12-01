import os, sys
from re import L
import cohere 
from cohere.classify import Example
sys.path.append(os.environ['PWD'])
from commune import BaseModule
from commune.utils import *
import pandas as pd
import numpy as np

class ClientModule(BaseModule):
    default_config_path =  'cohere.client.module'

    def __init__(self, config=None):
        BaseModule.__init__(self, config=config)
        try :
            self.get_json("cohere")
        except Exception as e: 
            print(e)
            self.put_json('cohere', [])
        self.api = self.config.get('api')
        self.models = self.config.get('baseline')
        self.co = cohere.Client(self.api)
        self.title = ""
        self.Examples = [item["title"] for item in self.get_json("cohere")]
        self.prompt = ""
        self.input_table = None
        

    def api_key(self):
        try:
            c = cohere.Client(self.api)
        except Exception:
            st.error("The API Key set to this state does not exist within co:here")
            return 
        st.success("Connected to co:here API ", icon="✅")


    def _test_(self):
        for state in st.session_state.items():
            st.write(state)
    
    def __pricing(self, inputs=[]):
        """
        Determine Price of the call to the Classify API 
            - Current Standard per 1000 calls
               Small  - $5
               Medium - $5
        """
        # given our current call count determine with the amount of inputs
        col1, col2 = st.columns([1,2])
        with col1: 
            if not "model" in st.session_state:
                st.session_state["model"] = ""
            if not "credit" in st.session_state:
                st.session_state["credit"] = self.config.get('credit')
            st.metric(label=f"{st.session_state['model']}($5/1000 queue) ", value=f"{len(inputs)}", delta=f"$ {(0.005*len(inputs))}")
        with col2: 
            st.metric(label=f"balance ", value=f"${st.session_state['credit']}", delta=f"-{(0.005*len(inputs))}")
        
       

    def __models(self):
        """
         Determine the models
            - Small
            - Medium
            - Large
            - Later (Custom)
        """
        model = st.selectbox("", self.models, label_visibility="collapsed")
        if not "model" in st.session_state:
            st.session_state["model"] = ""
        st.session_state["model"] = model


    def remove_json(self, title):
        cache = self.get_json("cohere")
        new_state = [item for item in cache if item["title"] != title]
        self.put_json("cohere", new_state)
        

    def __navagation(self):
        """
        Navagation tool to hold 
            - models
                - Example Models
            - buttons
                - export code
                - share
        """
        
        with st.sidebar:
            
            st.markdown("<h1 style='text-align: center;'>co:here SDK</h1>", unsafe_allow_html=True)
            with st.expander("✨ Models"):
                self.__models()
            with st.expander("📝 Presets"):
                with st.container():
                    st.header("Example Presets")
                    v = st.selectbox("", self.Examples, label_visibility="collapsed")
                    st.button("remove state", on_click=self.remove_json(v))


            with st.expander("💽 State"):
                st.json(st.session_state)

            with st.expander("⚙️ Setting"):
                if not "credit" in st.session_state:
                    st.session_state["credit"] = self.config.get('credit')  
                value = st.number_input(label="Credit", value=st.session_state["credit"])
                st.session_state["credit"] = value
                
    

    def __upload__(self):
        """
        Upload Button to import cvs to examples
        """
        if "examples" in st.session_state:
            st.session_state["examples"] = []

        upload = st.file_uploader("Choose a csv file", type="csv", accept_multiple_files=False)
        if upload:
            uploaded_data_read = pd.read_csv(upload)
            if len(uploaded_data_read.values) > 5:
                data = [(item[0], item[1]) for item in uploaded_data_read.values]
                st.dataframe(data, width=700, height=178)
                st.session_state["examples"] = data
            else:
                st.info("There needs to be 5 or more examples", icon="🤔")
                st.session_state["examples"] = []


                

    def __execute__(self):
        response = {}
        if not "title" in st.session_state:
            st.session_state["title"] = ""
        if not "prompts" in st.session_state:
            st.session_state["prompts"] = [] 
        if not "examples" in st.session_state:
            st.session_state["examples"] = [] 
        if not "model" in st.session_state:
            st.session_state["model"] = ""

        if st.session_state["title"] == "":
            st.info("If you want to execute you need to add a title name", icon="🛑")
            return 
        if not "credit" in st.session_state:
            st.session_state["credit"] = self.config.get('credit')  
        
        if len(st.session_state["prompts"]) == 0: 
            st.warning("You didn't enter a prompt")
            return 
        try:
            response = self.co.classify(inputs=[item for item in st.session_state["prompts"]],
                               model=st.session_state["model"],
                               examples=[Example(*ex) for ex in st.session_state["examples"]])
        except Exception as e:
            return st.error(f"Somthing went wrong {e}")
        if not "output" in st.session_state:
            st.session_state["output"] = {}
        st.session_state["credit"] =  st.session_state["credit"] - (0.005 * len(st.session_state["prompts"]))
        self._save()
        st.session_state["output"] = f"{response.classifications}"
        


    def _save(self):
        # st.success("Saved...", icon="✅")
        state = { key : value for key, value in st.session_state.items() if key!=""}
        cache = self.get_json("cohere")
        for item in cache:
            if "title" in item and item["title"] == state["title"]:
                st.warning("State exist")
                return
        cache.append(state)
        st.write(self.put_json("cohere", cache))
        


    def _clear(self):
        self.input_table = None
        self.Examples = []
        st.session_state["title"] = ""
        st.session_state["examples"] = []
        st.session_state["prompts"] = []


    def __streamlit__(self):
        """
        launcher for streamlit application
        """         
        
        st.markdown("<h1 style='text-align: center;'>🚀 Playground</h1>", unsafe_allow_html=True)

        st.session_state["title"] = st.text_input(label="✨Title✨", placeholder="Title Name", value=f"{self.title}")        
    
        with st.expander("Examples"):
            self.__upload__()

        with st.expander("Prompts"):
                
                self.prompt = st.text_input(label="", placeholder="Enter Prompt", label_visibility="collapsed")
                col1, col2 = st.columns([1,8])
   
                with col1:
                    if st.button("append"):
                        if not "prompts" in st.session_state:
                            st.session_state["prompts"] = []
                        if not self.prompt in st.session_state["prompts"] and len(st.session_state["prompts"]) < 32:
                            st.session_state["prompts"].append(self.prompt)


                    
                with col2:
                    if st.button("remove"):
                        if not "prompts" in st.session_state:
                            st.session_state["prompts"] = []
                        if "prompts" in st.session_state and self.prompt in st.session_state["prompts"]:
                            st.session_state["prompts"].remove(self.prompt)            
                
                if not "prompts" in st.session_state:
                    st.session_state["prompts"] = []   
                self.input_table = st.dataframe(pd.DataFrame(np.array(st.session_state["prompts"]),columns=['Inputs']), width=1000,height=175)
                self.__pricing(st.session_state["prompts"])                    

                if "prompts" in st.session_state and  len(st.session_state["prompts"]) == 32:
                    st.info("co:here api can not handel more then 32 inputs.")


        with st.expander("output"):
                if not "output" in st.session_state:
                    st.session_state["output"] = {}  
                st.write(st.session_state["output"])

        
        btn1, btn2, btn3 = st.columns([1.5,1.1,9])
        with btn1:
            st.button("Execute", on_click=self.__execute__)
            
        with btn2:
            st.button("save", on_click=self._save)

        with btn3:
            st.button("clear all", on_click=self._clear)

        self.__navagation()
        # self._test_()

if __name__ == "__main__":
    import streamlit as st
    ClientModule().__streamlit__()