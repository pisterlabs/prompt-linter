import streamlit as st

def info_page(): 
    st.info('This project is currently in development and bugs can be reported here : [Github issue](https://github.com/americium-241/Omnitool_UI/tree/master)')

    st.markdown("""
                # Usage guidelines
                
                ### API keys : 
                   
                - Get an openAI key at : [OpenAI](https://platform.openai.com/)
                - Hugging face key can be ommited if related tools are not used, or can befound at : [HuggingFace](https://huggingface.co/docs/hub/security-tokens)
                - Keys can be added simply to the app (see custom)
            
                
                ## Set-up the chatbot in the settings page :

                - Enter your API keys 
                - Choose a model : OpenAI or Llama are available (see custom)
                - Pick an agent : How is the chatbot supposed to behave ? 

                    All agents are very different, try and explore, but so far the OpenAI one does handle tools the most reliably (see custom)
                    be careful when using gpt-4 as token number and price can escalade rather quickly. 
                
                - Define prefix and suffix depending on the type of session you want to initiate

                    These are added at the beginning and end of the user input 
                
                - Load pdf or txt files to the vector database for similarity search in documents. 

                    Relevant document chunk are added to the chatbot prompt before the user input
                
                - Load any document to the workspace to facilitate future use of tools for in chat data manipulation

                - Try the vocal control, this thing holds with strings so maybe it will crack, but never miss
                    a chance to say hello to Jarvis.
                
                ## Select tools in the tool page : 

                - Filter tool by name and description 
                - Select tools card and options 
                - In app add tool at the end of cards list : 

                    * Name the python file to be created
                    * Write a function (single arguments works best for all agents)
                    * add a docstring 
                    * add a relevant return that is sent to the chatbot
                    * submit and use 


                ## Discuss with chatbot in the chat page: 

                - Start the session and ask a question, or select a previous session and continue it
                - The bot can usually handle itself the tool calls, but results are more reliable with explicit usage description. For complex actions you should precise the tools execution order
                - Change tools, settings, come back and explore multiple configuration within one session
            
                ## Custom
                
                ### Tools

                You can make custom tools from multiple ways : 

                1. Make a new python file at Omnitool_UI/tools/tools_list : 
            
                - make a single function with a docstring to describe tool usage and a return relevant for the chatbot though process""")
    function_exemple = """
                        import streamlit as st 
                
                        def streamlit_info(message):
                            ''' This function displays the message as a streamlit info card'''
                            st.info(message)
                            return 'Success '
                            """
    st.code(function_exemple,language="python")            
    st.markdown("""
                    - make a single class that inherits from UI_Tool with a _run method and a _ui method for option management
                            The TestTool option can guide you to the absolute path of the folder
                               
                    """)

    tool_exemple =""" 
                import streamlit as st
                from streamlit_elements import elements, mui, html
                import os 
                from storage.logger_config import logger
                from tools.base_tools import Ui_Tool


                Local_dir=dir_path = os.path.dirname(os.path.realpath(__file__))

                class Testtool(Ui_Tool):
                    name = 'Testtool'
                    icon = '🌍'
                    title = 'Test tool'
                    description =  'This function is used so the human can make test, thank you to proceed, input : anything'

                    def _run(self, a):
                        # This function is executed by the chatbot when using tool
                        st.success(a)
                        logger.debug('During the test tool execution and with input : ' + a)
                        return 'Success'

                    def _ui(self):
                        # This function is executed at the creation of the tool card in the tool page
                        if "test_state" not in st.session_state: 
                            st.session_state.test_state = False

                        def checkstate(value):
                            st.session_state.test_state = value['target']['checked']
                            if st.session_state.test_state is True : 
                                st.success('Find me at '+ Local_dir)

                    # Expander placed outside (below) the card
                        with mui.Accordion():
                            with mui.AccordionSummary(expandIcon=mui.icon.ExpandMore):
                                mui.Typography("Options")
                            with mui.AccordionDetails():
                                mui.FormControlLabel(
                                    control=mui.Checkbox(onChange=checkstate,checked= st.session_state.test_state),
                                    label="Try to change me !")"""

    st.code(tool_exemple, language="python")
                            
                    
    st.markdown("""
                
                2. Use the in-app add tool form. Only supports function tool creation
                3. Use the chatbot make_tool tool. Only supports function tool creation
                any tool create by the form or the make_tool are creating a new tool file (Omnitool_UI/tools/tools_list)
                    
                ### Agents

                You can make custom agents by creating a new python file at Omnitool_UI/agents/agents_list : 

                - Write a single class with an initialize_agent method that returns an object with a run method. The output of the run is expected to be the answer to the user input 
                - The custom agent example, taken from langchain how to, gives a minimalistic template to begin
                
                ### API keys 

                API keys are accessible in the config file. New text inputs can be added to the app simply by extending the KEYS list. 
                This is useful to set up the environment necessary for the execution of your tools

                ### Config file

                - Other parameters can be modified in the config file : 

                    - Models list
                    - Agents list 
                    - Vector db chunk size embedding and number of document retrieved per similarity search
                    - Voice command time_outs
                    - Maximum intermediate thoughts iteration
                    - Logging level 

                - Thanks to streamlit interactivity, all files can be modified during the app execution that will continue to work and run the new code at next trigger

                ## Troubleshooting

                This project is in development and bugs are to be expected. The flexibility of streamlit can lead to dead ends when combined with cached data (at our stage at least), sometimes a simple refresh is your best call.
                Bug can be reported at : 
                Also available in right side menu 

                
                    """)