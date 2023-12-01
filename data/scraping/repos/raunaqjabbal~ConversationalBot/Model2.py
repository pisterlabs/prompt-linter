# First
import openai 
import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory   
from langchain.prompts import PromptTemplate
import os
from langchain.chat_models import ChatOpenAI

import json
import time
# os.environ["OPENAI_API_KEY"]="" 

k=0

st.set_page_config(
        page_title="AI Assistant",
)


if not os.path.exists("data"):
    os.makedirs("data")

# template = "You need to be VERY persuasive and friendly with me. Convince me to tell my name, email, phone number, address, date of birth, and education "

prompt2 = ChatPromptTemplate.from_messages([
        SystemMessage(content="""Only extract VALID personal details like name, email address, phone number, address, date of birth, and educational details from the conversation below. 
Do not accept a portion of the email address and it should have domain name like gmail.com. Phone number should have correct number of digits. 
Reply in json format only, for any missing values fill with NA. Make sure that the dates for the birthday are correct. Birth years after 2023 are invalid. Do not generate values from the below text:\n"""), 
        HumanMessagePromptTemplate.from_template( template = "{human_input}")
        ])

@st.cache_resource
def create():
    st.title("💬 Conversational Chatbot") 
    st.session_state["messages"] = [{"role": "assistant", "content": "Please provide me your details"}]
    
    llm = ChatOpenAI(temperature=1)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=
                    """Pretend to be AI in this story who is Humans friend. Convince Human to tell his name, email, phone number, address, date of birth, and education and other personal information. 
    You need to be very persuasive and ASK FOR INFORMATION from Human. Do not accept a portion of the email address and it should have domain name like @gmail.com. 
    Make sure that the dates for the birthday are correct. Birth years after 2023 are invalid. Write a 1 line reply"""), 
        MessagesPlaceholder(variable_name="chat_history"), 
        HumanMessagePromptTemplate.from_template(template = "{human_input}")
    ])
        
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # memory = ConversationBufferMemory()
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )
    return memory, chain, llm

memory, chain, llm= create()

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, Please let me know your details!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if reply := st.chat_input():
    if k==0:
        st.session_state.messages.append({"role": "user", "content": reply})
        st.chat_message("user").write(reply)
        demo = chain.predict(human_input=reply)
        msg = {
        "role": "assistant",
        "content": demo
        }
        st.session_state.messages.append(msg)
        st.chat_message("assistant").write(msg['content'])
    if k==1:
        st.error("Conversation has ended.")
    
    
    
if st.sidebar.button("End Conversation", key="delfiles"):
    output = ""
    
    
    print()
    for idx,i in enumerate(memory.chat_memory.messages):
        if idx%2==0:
            output+=i.content+"\n"
        
    if output=="":
        ans="No text provided"
        
        st.sidebar.error(ans)
    else:
        
        ans = prompt2.format(human_input=output)
        print(ans)
        ans = llm.predict(ans).replace("\n","")
        st.sidebar.write(ans)
        
        
        with open(f"data/{time.time()}.json", "w") as outfile:
            outfile.write(ans)
    st.stop()
    st.session_state.stop()
    k = 1
    memory.clear()
    