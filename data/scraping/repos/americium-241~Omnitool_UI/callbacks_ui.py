import streamlit as st 
from langchain.callbacks.base import BaseCallbackHandler
from storage.logger_config import logger


class Custom_chat_callback(BaseCallbackHandler):

    def on_llm_new_token(self, token: str, **kwargs) -> None:
       st.session_state.token_count=st.session_state.token_count+1
       st.session_state.all_tokens=st.session_state.all_tokens+str(token)
      
    def on_llm_start(self, serialized,prompts, **kwargs):
        """Run when LLM starts running."""

    def on_chat_model_start(self, serialized, messages, **kwargs):
        """Run when Chat Model starts running."""
        #st.sidebar.info('Chat begins')

    def on_llm_end(self, response, **kwargs):
        """Run when LLM ends running."""
        logger.info('-- llm tokens -- :'+str(st.session_state.token_count ))
        with st.session_state.token_count_placeholder:
             st.info('Tokens count : '+str(st.session_state.token_count ))

    def on_llm_error( self, errors, **kwargs):
        """Run when LLM errors."""
        logger.debug('on_llm_error'+str(errors))
        
    
class ToolCallback(BaseCallbackHandler):

    def should_check(self,serialized_obj: dict) -> bool:
        # Define condition and call it on dedicated callback
        return serialized_obj.get("name") == "Code_sender"

    def on_tool_start(self,serialized,input_str, **kwargs) -> None:
        """Run when tool starts running."""
        if self.should_check(serialized) :
            logger.info('Tool started')
            #Add a custom handling of the tool like a human confirmation for instance

    def on_tool_end(self,output,**kwargs) -> None:
        """Run when tool ends running."""
        logger.info('Tool ended')

    def on_tool_error(self,error,**kwargs) -> None:
        """Run when tool errors."""
        logger.debug('Tool failed and broke into pieces')



   
        



