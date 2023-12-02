import time

import streamlit as st
from friend_replica.format_chat import ChatConfig, format_chat_history, split_chat_data
from friend_replica.recollection import LanguageModelwithRecollection
from friend_replica.semantic_search import *
from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate
from models.model_cn import ChatGLM

if st.session_state.language == 'chinese':
    model = ChatGLM()
else: 
    model = GPT4All(model="/home/enoshima/workspace/intel/models/llama-2-7b-chat.ggmlv3.q4_0.bin")
    
chat_config = ChatConfig(
    my_name=st.session_state.my_name,
    friend_name=st.session_state.friend_name,
    language=st.session_state.language
)
chat_with_friend = Chat(device='cpu', chat_config=chat_config)
m = LanguageModelwithRecollection(model, chat_with_friend)
chat_blocks = split_chat_data(chat_with_friend.chat_data)

### Side Bar Module ###
with st.sidebar:
    "[Get a Comma API key](https://github.com/roxie-zhang/friend_replica)"
    "[View the source code](https://github.com/roxie-zhang/friend_replica)"
    
st.title("Comma Friend Replica")
st.caption("🚀 Chat with your friend! "
           "| *FDU Comma Team Ver-1.1*")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "current_chat" not in st.session_state:
    st.session_state.current_chat = []
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if new_msg := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(new_msg)
    st.session_state.messages.append({"role": "user", "content": new_msg})
    st.session_state.current_chat_replica.append(chat_config.friend_name + ': ' + new_msg)
    
    with st.chat_message("assistant"):
        thoughts, key_words = m.generate_thoughts(new_msg)
        
        if isinstance(thoughts[0], list):
            recollections = ['\n'.join(format_chat_history(thought, chat_config, for_read=True, time=True)) for thought in thoughts]
        else:
            recollections = ''
        
        st.markdown(f'概括关键词：{key_words}' if st.session_state.language == 'chinese' else f'Summarizing message as:{key_words}')
        st.session_state.messages.append({"role": "assistant", "content": f'概括关键词：{key_words}' if st.session_state.language == 'chinese' else f'Summarizing message as:{key_words}'})
        
        if chat_config.language == "english":
            template = """[[INST]]<<SYS>>Please tell me when the following conversation took place, and
            summarize its main idea into only one sentence with regard to {key_words}: 
            <</SYS>>
            
            {recollections}

            One-sentence summary:
            [[/INST]] """

        else:
            template = """请告诉我下列对话的发生时间，并用一句话简短地概括它的整体内容，其中关键词为 {key_words}：
            
            [Round 1]
            对话：
            2023-08-16T11:33:44 from friend: 中午去哪吃？
            2023-08-16T11:35:14 from me: 西域美食吃吗
            2023-08-16T11:33:44 from friend: 西域美食
            2023-08-16T11:33:44 from friend: 好油啊
            2023-08-16T11:33:44 from friend: 想吃点好的
            2023-08-16T11:35:14 from me: 那要不去万达那边？
            2023-08-16T11:33:44 from friend: 行的行的
            
            总结：
            以上对话发生在2023年8月16日中午，我和我的朋友在商量中饭去哪里吃，经过商量后决定去万达。
            
            [Round 2]
            对话：
            {recollections}
            
            总结："""
            
        prompt = PromptTemplate(
            template=template, 
            input_variables=[
                'key_words',
                'recollections',
            ],
        )
        
        out = []
        for recollection in recollections:
            prompt_text = prompt.format(key_words=key_words, 
                                        recollections=recollection,
                                        )
            if chat_config.language == "english":
                out0 = model(prompt_text).strip()
                st.markdown(f'Recollected following conversation: \n{recollection}')
                st.session_state.messages.append({"role": "assistant", "content": f'Recollected following conversation: \n{recollection}'})
                st.markdown(f'Summary: \n{out0}')
                st.session_state.messages.append({"role": "assistant", "content": f'Summary: \n{out0}'})

            else:
                out0 = model(prompt_text)[len(prompt_text):].strip()
                st.markdown(f'回忆以下对话：\n{recollection}')
                st.session_state.messages.append({"role": "assistant", "content": f'回忆以下对话：\n{recollection}'})
                st.markdown(f'概括：\n{out0}')
                st.session_state.messages.append({"role": "assistant", "content": f'概括：\n{out0}'})
            out.append(out0)
        
        if chat_config.language == "english":
            prompt_template = """[[INST]]<<SYS>>You are roleplaying a robot with the personality of {my_name} in a casual online chat with {friend_name}.
            Refer to Memory as well as Recent Conversation , respond to the latest message of {friend_name} with one sentence only.
            Start the short, casual response with {my_name}: 
            <</SYS>>
            
            Memory:
            '''
            {recollections}
            '''

            Recent Conversation:
            '''
            {recent_chat}
            '''

            {current_chat}
            [[/INST]] """
            
        else:
            prompt_template = """接下来请你扮演一个在一场随性的网络聊天中拥有{my_name}性格特征的角色。
            首先从过往聊天记录中，学习总结{my_name}的性格特点，并掌握{my_name}和{friend_name}之间的人际关系。
            之后，运用近期聊天内容以及记忆中的信息，回复{friend_name}发送的消息。
            请用一句话，通过简短、随意的方式用{my_name}的身份进行回复：
            
            记忆：
            '''
            {recollections}
            '''

            近期聊天：
            '''
            {recent_chat}
            '''
 

            {current_chat}
            
            """
        
        prompt = PromptTemplate(
            template=prompt_template, 
            input_variables=[
                'my_name', 
                'friend_name', 
                'recent_chat', 
                'recollections',
                'current_chat'
            ],
        )
        
        prompt_text = prompt.format(
            my_name=chat_config.my_name,
            friend_name=chat_config.friend_name,
            recent_chat='\n'.join(format_chat_history(chat_blocks[-1], chat_config, for_read=True)),
            recollections=recollections,
            current_chat='\n'.join(st.session_state.current_chat_replica)
        )
        
        if chat_config.language == "english":
            response = model(prompt_text, stop='\n')
        else:
            response = model(prompt_text)[len(prompt_text):].split('\n')[0]
            
        st.markdown(response)
        st.session_state.current_chat_replica.append(response)
        st.session_state.messages.append({"role": "assistant", "content": response})