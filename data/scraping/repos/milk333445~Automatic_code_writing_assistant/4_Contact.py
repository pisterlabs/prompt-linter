import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, OpenAI, LLMChain
import time
import os
import re
from typing import List
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage
)
class CAMELAgent:
    def __init__(
        self,
        system_message: SystemMessage,
        model: ChatOpenAI,
        max_messages: int = 10000,
    ) -> None:
        self.system_message = system_message
        self.model = model
        self.max_messages = max_messages
        self.init_messages()
    
    def reset(self) -> None:
        self.init_messages()
        return self.stored_messages
    
    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]
        
    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        while len(self.stored_messages) > self.max_messages:
            # Remove the oldest HumanMessage or AIMessage
            self.stored_messages.pop(1)
        return self.stored_messages
    
    def step(
        self,
        input_message: HumanMessage,
    ) -> AIMessage:
          messages = self.update_messages(input_message)
          
          output_message = self.model(messages)
          
          self.update_messages(output_message)
          
          return output_message

def get_sys_msgs(assistant_role_name: str, assistant_inception_prompt, main_task, file_list_str, code_history_str, data_api_design):
    assistant_sys_template = SystemMessagePromptTemplate.from_template(template=assistant_inception_prompt)
    assistant_sys_msg = assistant_sys_template.format_messages(assistant_role_name=assistant_role_name, 
                                                               main_task = main_task,
                                                               file_list_str = file_list_str,
                                                               code_history_str = code_history_str,
                                                               data_api_design = data_api_design)[0]
    return assistant_sys_msg
def show_code(code_history):
    if not code_history:
        return ""
    code_history_str = ""
    for snippet in code_history:
        code_history_str += f"## {snippet.title}\n"
        code_history_str += f"```python\n{snippet.code}\n```\n"
    return code_history_str   

st.title("Code Assistant")
#初始化
if "messages" not in st.session_state:
    st.session_state.messages = []

#把之前的對話紀錄在重整時顯現
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])



with st.sidebar:
    
    openai_api_key = st.text_input('OpenAI API Key', '', type="password")
    st.info("如果您有任何與任務相關問題，請在下方輸入，我會盡力回答您。")

os.environ['OPENAI_API_KEY'] = openai_api_key

try:
    code_history = st.session_state["code_history"]
    main_task = st.session_state["main_task"]
    data_api_design = st.session_state["data_api_design"]
    python_package_name = st.session_state["python_package_name"]
    seq_flow = st.session_state["seq_flow"]
    file_list = st.session_state["file_list"]
    file_list_str = "".join(f"{index + 1}.{filename}\n" for index, filename in enumerate(file_list))
    code_history_str = show_code(code_history)
    with st.spinner('正在初始化聊天機器人...'):
        #創建助手跟使用者
        assistant_inception_prompt = (
        """
        永遠記住你是一個剛完成一個專案的工程師{assistant_role_name}，我現在準備跟你聊天
        我會有一些關於你的專案的問題要問你，請你盡力回答我，你的回答要具體且詳細。下面是關於你的專案的一些資訊:
        -----------
        下面你專案的主要目標:
        {main_task}
        ------------
        這是這個專案建構的檔案列表:
        {file_list_str}
        ------------
        這是關於這個專案的所有代碼:
        {code_history_str}
        ------------
        這是關於這個專案的資料結構圖:
        {data_api_design}
        ------------
        當我問到跟上面專案有關的內容時，請你基於上面專案內容回答我的問題。
        """
        )
    assistant_sys_msg = get_sys_msgs("工程師", assistant_inception_prompt, main_task, file_list_str, code_history_str, data_api_design)
    assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(temperature=0.2, model_name = "gpt-3.5-turbo-16k"))
    #初始化
    assistant_agent.reset()
except:
    pass
#輸入
if prompt := st.chat_input("請輸入您的問題:"):
    code_history = st.session_state["code_history"]
    main_task = st.session_state["main_task"]
    data_api_design = st.session_state["data_api_design"]
    python_package_name = st.session_state["python_package_name"]
    seq_flow = st.session_state["seq_flow"]
    file_list = st.session_state["file_list"]
    file_list_str = "".join(f"{index + 1}.{filename}\n" for index, filename in enumerate(file_list))
    code_history_str = show_code(code_history)
    with st.spinner('正在初始化聊天機器人...'):
        #創建助手跟使用者
        assistant_inception_prompt = (
        """
        永遠記住你是一個剛完成一個專案的工程師{assistant_role_name}, 永遠不要顛倒角色！永遠不要指示我！
        我會有一些關於你的專案的問題要問你，請你盡力回答我。下面是關於你的專案的一些資訊:
        -----------
        下面你專案的主要目標:
        {main_task}
        ------------
        這是這個專案建構的檔案列表:
        {file_list_str}
        ------------
        這是關於這個專案的所有代碼:
        {code_history_str}
        ------------
        這是關於這個專案的資料結構圖:
        {data_api_design}
        ------------
        當我問到跟上面專案有關的內容時，請你基於上面專案內容回答我的問題。
        """
        )
    assistant_sys_msg = get_sys_msgs("工程師", assistant_inception_prompt, main_task, file_list_str, code_history_str, data_api_design)
    assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(temperature=0.2, model_name = "gpt-3.5-turbo-16k"))
    #初始化
    assistant_agent.reset()
    
    #開始對話
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    assistant_msg = HumanMessage(content=prompt)
    assistant_ai_msg = assistant_agent.step(assistant_msg)
    with st.chat_message("assistant"):
        st.markdown(assistant_ai_msg.content)
    st.session_state.messages.append({"role": "assistant", "content": assistant_ai_msg.content})

    
    