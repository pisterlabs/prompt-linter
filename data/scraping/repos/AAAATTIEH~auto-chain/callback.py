from langchain.callbacks.base import BaseCallbackHandler
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union
from langchain.schema.messages import BaseMessage
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.schema.document import Document
from langchain.schema.messages import BaseMessage
from langchain.schema.output import LLMResult
from utils.session_state import *
import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import copy
from models.llms.llms import *
from .document import DocumentProcessor
from .table_parse import get_table
import json
import os
from io import StringIO
RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
from utils.multi_modal import st_multi_modal

def is_substring(input_string, word):
    # Convert both the input_string and word to lowercase for case-insensitive comparison
    input_string = input_string.lower()
    word = word.lower()

    # Iterate through all possible substrings of the word
    for i in range(len(word)):
            # Check if the substring of the word is equal to the input_string
           
            if word[0:i+1] == input_string:
                return True
    if(input_string.startswith(word)):
         return True
    # If no match is found, return False
    return False


def prettify_output(input_string):
    lines = input_string.split('\n')
    last_line = lines[-1]
    last_line = last_line.strip()
    lines = lines[:-1]
    output = ''
    for line in lines:
        line = line.strip()
        if(line.startswith('Thought:')):
            output+= line[8:]
        elif(line.startswith('Action:')):
            pass
        elif(line.startswith('Action Input:')):
            pass
        elif(line.startswith('I now know')):
            pass
        elif(line.startswith('Final Answer:')):
            output+=line[13:]
        else:
            output+=line
        output+='  \n'
    if(is_substring(last_line,"Thought:")):
        if(last_line.startswith("Thought:")):
            output += last_line[8:]
        else:
            pass
    elif(is_substring(last_line,"Action:")):
        pass
    elif(is_substring(last_line,"Action Input:")):
        pass
    elif(is_substring(last_line,"I now know")):
        if(last_line.startswith("I now know")):
            pass
        else:
            pass
    elif(is_substring(last_line,"Final Answer:")):
        if(last_line.startswith("Final Answer:")):
            output+=last_line[13:]
        else:
            pass
    else:
        output+=last_line
    return output    
class CustomHandler(BaseCallbackHandler):
    
    """Base callback handler that can be used to handle callbacks from langchain."""
    def __init__(self,message_placeholder):
        self.message_placeholder = message_placeholder
        self.content = ''
        self.test_token = ''
        self.containers = []
    SPACES = '>'
    text_container = None
    stack = []
    source_documents  = []
    add = True
    def write_output(
            self,text
    ):
        
        if not self.text_container:
            self.text_container =st.expander("Thought Process",expanded=False)
            if(executor_session_state().memory is not None):
                messages_session_state()[-1]["thought_process"] = "**Memory**\n\n"

                messages_session_state()[-1]["thought_process"] += "\n\n".join([f"{type(item).__name__}: {item}" for item in executor_session_state().memory.chat_memory.messages])
            else:
                messages_session_state()[-1]["thought_process"] = ''
        with self.text_container:
                st.markdown(self.SPACES+' '+text)
        messages_session_state()[-1]["thought_process"] += '\n\n'+self.SPACES+' '+text
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        self.add = True
        output = f':green[**Start LLM**]'
        self.stack.append('LLM')
        self.write_output(output)
        self.SPACES+='>'
        self.test_token = ''

    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        
        """Run when Chat Model starts running."""
        output = f':green[**Start LLM**]'
        self.stack.append('LLM')
        self.write_output(output)
        #json_str = json.dumps(serialized, indent=4)  
        #markdown_text = f"```json\n{json_str}\n```"
        #self.write_output(markdown_text) 
        self.SPACES+='>'
    def on_retry(self,*args,**kwargs):
        """Run on retry""" 

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:

        """Run on new LLM token. Only available when streaming is enabled."""
        try:
            #Check the last line
            self.content+=token
            output = prettify_output(self.content)
            messages_session_state()[-1]["content"] = output
            self.containers = st_multi_modal(container=self.message_placeholder,input_string=output+"▌",subcontainers=self.containers)
                #self.message_placeholder.write(replaced+ )
        except:
            pass
     
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        self.content+='\n'

        self.SPACES = self.SPACES[:-1]
        output = f':green[**End {self.stack[-1]}**]'
        self.stack.pop()
        self.write_output(output)
        
    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        
        """Run when LLM errors."""


    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        
        """Run when chain starts running."""
        class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        output = f':blue[**Entering New {class_name} Chain**]'
        self.stack.append(f'{class_name} Chain')
        self.write_output(output)
        self.SPACES+='>'
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
  
        self.SPACES = self.SPACES[:-1]
        output = f':blue[**Finished {self.stack[-1]}**]'
        self.stack.pop()
        self.write_output(output)
        if 'source_documents' in outputs:
            self.source_documents.extend(outputs['source_documents'])
    
        #if 'intermediate_steps' in outputs:
        #    documents = outputs['intermediate_steps'][-1][-1]
        #    self.source_documents.extend(documents)
    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        
        output = f':orange[**Tool START {serialized["name"]}**]'
        self.stack.append(f'{serialized["name"]} Tool')
        self.write_output(output)
        self.SPACES+='>'


    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        self.SPACES = self.SPACES[:-1]
        
        outputy = f':orange[**End {self.stack[-1]}**]'
        self.stack.pop()
        self.write_output(outputy)
        self.write_output(output)

        if kwargs['name'] == 'search':
            pass
        elif 'Error' in output and kwargs['name']=='python_repl_ast':
            self.content = self.content +':red[An Error Happened]'+'\n'
        elif 'AxesSubplot' in output:
            fig = plt.gcf()
            path = f'dataset/process/{st.session_state.user_id}/{st.session_state.session_id}/output/images/image{st.session_state.model["index"]}.png'
            dir = f'dataset/process/{st.session_state.user_id}/{st.session_state.session_id}/output/images'
            if(not os.path.exists(dir)):
                os.makedirs(dir)
            st.session_state.model["index"]+=1
            fig.savefig(path)
            plt.clf() 
            self.content = self.content +'<{"type":"chart","source" : "'+path+'"}>'+'\n'
        elif f'dataset/process/{st.session_state.user_id}/{st.session_state.session_id}/input/images' in output:
            self.content = self.content +'<{"type":"image","source" : "'+output.strip()+'"}>'+'\n'
        else:
            tables = get_table(output)
            if not tables:
                self.content = self.content +output.strip()+'\n'
            else:
                for table in tables:
                    df = pd.DataFrame(table)
                    dir = f'dataset/process/{st.session_state.user_id}/{st.session_state.session_id}/output/tables'
                    if(not os.path.exists(dir)):
                        os.makedirs(dir)
                    path = f'dataset/process/{st.session_state.user_id}/{st.session_state.session_id}/output/tables/table{st.session_state.model["index"]}.csv'
                    df.to_csv(path,index=False)
                    st.session_state.model["index"]+= 1
                    self.content = self.content +'<{"type":"table","source" : "'+path+'"}>'+'\n'
        
        
        output = prettify_output(self.content)
        messages_session_state()[-1]["content"] = output
        self.containers = st_multi_modal(container=self.message_placeholder,input_string=output+"▌",subcontainers=self.containers)
    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""


    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""
        #self.write_output(text)
    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        splitted_strings = action.log.split('\n')

        formatted_strings = [f":violet[**{s}**]" for s in splitted_strings if s.strip()]
        for s in formatted_strings:
            if s != '****':
                self.write_output(s)

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        output = prettify_output(self.content)
        
        messages_session_state()[-1]["content"] = output
        self.containers = st_multi_modal(container=self.message_placeholder,input_string=output,subcontainers=self.containers)
        self.containers = ""
        self.content = ''
        self.write_output(finish.log)
        self.text_container.expanded = False
        
        #processor = DocumentProcessor(self.source_documents)   
        #grouped_documents = processor.group_and_sort()
        #messages_session_state()["source_documents"] = grouped_documents
        
        self.source_documents.clear()

