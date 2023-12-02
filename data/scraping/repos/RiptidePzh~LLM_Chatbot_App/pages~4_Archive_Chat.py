import datetime
import time

import streamlit as st
from friend_replica.format_chat import ChatConfig, format_chat_history, split_chat_data
from friend_replica.recollection import LanguageModelwithRecollection
from friend_replica.semantic_search import *
from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate
from model_paths import path_en
from models.model_cn import ChatGLM

with st.sidebar:
    "[Get a Comma API key](https://github.com/roxie-zhang/friend_replica)"
    "[View the source code](https://github.com/roxie-zhang/friend_replica)"

### Header Module ###
st.title("Comma Friend Replica - Archive Chat")
st.caption("🚀 Chat with your friend base on preprocessed memory archive! "
           "| *FDU Comma Team Ver-1.1*")

if "messages_archive" not in st.session_state:
    st.session_state.messages_archive = []

if st.session_state.language == 'chinese':
    model = ChatGLM()
else: 
    model = GPT4All(model=path_en)
    
chat_config = ChatConfig(
    my_name=st.session_state.my_name,
    friend_name=st.session_state.friend_name,
    language=st.session_state.language
)
chat_with_friend = Chat(device='cpu', chat_config=chat_config)
m = LanguageModelwithRecollection(model, chat_with_friend, debug=True)
chat_blocks = split_chat_data(chat_with_friend.chat_data)

# Load Personality Archive
personality_archive = os.path.join(m.chat.friend_path, f'personality_{m.chat.chat_config.friend_name}.json')
if os.path.exists(personality_archive):
    with open(personality_archive,'r', encoding='utf-8') as json_file:
        personality_archive = json.load(json_file)
else:
    # Initialize Personality Archive if not initialized before
    personality_archive = m.personality_archive()

# Load Memory Archive
memory_archive = os.path.join(m.chat.friend_path, f'memory_{m.chat.chat_config.friend_name}.json')
if os.path.exists(memory_archive):
    with open(memory_archive,'r', encoding='utf-8') as json_file:
        memory_archive = json.load(json_file)
else:
    # Initialize Memory Archive if not initialized before
    memory_archive = m.memory_archive()

with st.chat_message('assistant'):
    auto_reply = f"Hi, {m.chat.chat_config.friend_name}! I'm the agent bot of {m.chat.chat_config.my_name}. I have memory of us discussing these topics:\n"
    st.markdown(auto_reply)
    for i, memory_entry in enumerate(memory_archive):
        str_time = datetime.datetime.fromtimestamp(memory_entry['time_interval'][1]).strftime('%m.%d')
        st.markdown(f"#{i} {str_time}: {memory_entry['key_word']}\n")
    st.markdown("Do you want to continue on any of these?")


if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages_archive:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if input_index := st.chat_input("Enter the # of the topic if you want to continue: "):
    with st.chat_message("user"):
        st.markdown(input_index)
    st.session_state.messages_archive.append({"role": "user", "content": input_index})
    st.session_state.current_chat_archieve.append(chat_config.friend_name + ': ' + input_index)
        
    if input_index.isdigit():
        input_index = int(input_index)
        st.session_state.current_idx = input_index
        if input_index < len(memory_archive):
            with st.chat_message('assistant'):
                st.markdown(f"Okay! Let's continue on [{memory_archive[input_index]['key_word']}]\n" )
                st.session_state.messages_archive.append({"role": "assistant", "content": f"Okay! Let's continue on [{memory_archive[input_index]['key_word']}]\n" })
                memory = memory_archive[input_index]['memory']
                st.markdown("I recall last time: " + memory)
                st.session_state.messages_archive.append({"role": "assistant", "content": "I recall last time: " + memory})
                st.markdown("What do you think?")
                st.session_state.messages_archive.append({"role": "assistant", "content": "What do you think?"})
            st.session_state.continue_chat = True
            
    elif st.session_state.continue_chat:
        memory = memory_archive[st.session_state.current_idx]['memory']
        #assert len(chat_blocks) == len(memory_archive) and len(chat_blocks) == len(personality_archive)
        matching_chat_block = chat_blocks[st.session_state.current_idx]
        personality = personality_archive[st.session_state.current_idx]['personality']
        
        if m.chat.chat_config.language == "english":
            prompt_template = """[[INST]]<<SYS>>You are roleplaying a robot with the personality of {my_name} in a casual online chat with {friend_name}.
            as described here: {personality}.
            Refer to Memory as well as Recent Conversation , respond to the latest message of {friend_name} with one sentence only.
            Start the short, casual response with {my_name}: 
            <</SYS>>
                    
            Memory:
            '''
            {memory}
            '''

            Recent Conversation:
            '''
            {recent_chat}
            '''
            
            {current_chat}
            [[/INST]] """
            
        else:
            prompt_template = """接下来请你扮演一个在一场随性的网络聊天中拥有{my_name}性格特征的角色。
            首先从过往聊天记录中，根据{my_name}的性格特点{personality}，掌握{my_name}和{friend_name}之间的人际关系。
            之后，运用近期聊天内容以及记忆中的信息，回复{friend_name}发送的消息。
            请用一句话，通过简短、随意的方式用{my_name}的身份进行回复：
            
            记忆：
            '''
            {memory}
            '''

            近期聊天：
            '''
            {recent_chat}
            '''
 

            {current_chat}
            
            """


        prompt_text = prompt_template.format(
            my_name=m.chat.chat_config.my_name,
            friend_name=m.chat.chat_config.friend_name,
            personality=personality,
            memory=memory,
            recent_chat='\n'.join(format_chat_history(matching_chat_block, m.chat.chat_config, for_read=True)),
            current_chat='\n'.join(st.session_state.current_chat_archieve),
        )
            
        if m.chat.chat_config.language == "english":
            out = m.model(prompt_text, stop='\n').replace('\"', '').replace('�', '')
        else:
            out = m.model(prompt_text)[len(prompt_text):].split('\n')[0]
        
        st.session_state.current_chat_archieve.append(out)
        with st.chat_message('assistant'):
            st.markdown(out.split(':')[-1])
        st.session_state.messages_archive.append({'role': 'assistant', 'content': out.split(':')[-1]})
    
    else:
        with st.chat_message('assistant'):
            out = m.chat_with_recollection(input_index)
            st.markdown(out)
        st.session_state.current_chat_archieve.append(out)