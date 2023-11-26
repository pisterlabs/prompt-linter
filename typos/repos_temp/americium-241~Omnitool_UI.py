"""Question: {question}

        Answer: Let's think step by step.""""""You should create a python code that precisely solves the problem asked. Always make one single python snippet and assume that exemples should be made with randomly generated data rather than loaded ones.
    format : The python code should be formated as ```python \n ... \n ``` 
    ALWAYS finish your answer by \n TERMINATE""""""
                        import streamlit as st 
                
                        def streamlit_info(message):
                            ''' This function displays the message as a streamlit info card'''
                            st.info(message)
                            return 'Success '
                            """""" 
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