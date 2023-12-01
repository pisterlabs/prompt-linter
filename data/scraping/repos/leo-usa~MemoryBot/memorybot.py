"""
This is a Python script that serves as a frontend for a conversational AI model built with the `langchain` and `llms` libraries.
The code creates a web application using Streamlit, a Python library for building interactive web apps.
"""

# Import necessary libraries
import streamlit as st
from PIL import Image
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import re

def is_four_digit_number(string):
    pattern = r'^\d{4}$'  # Matches exactly four digits
    return bool(re.match(pattern, string))


# Set Streamlit page configuration
im = Image.open('sricon.png')
st.set_page_config(page_title=' 🤖ChatGPT with Memory🧠', layout='wide', page_icon = im)
# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []
if "just_sent" not in st.session_state:
    st.session_state["just_sent"] = False
if "temp" not in st.session_state:
    st.session_state["temp"] = ""
if "balance" not in st.session_state:
    st.session_state["balance"] = 0.0
if "deposit" not in st.session_state:
    st.session_state["deposit"] = 3.0

def clear_text():
    st.session_state["temp"] = st.session_state["input"]
    st.session_state["input"] = ""


# Define function to get user input
def get_text():
    """
    Get the user input text.

    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input("You: ", st.session_state["input"], key="input", 
                            placeholder="Your AI assistant here! Ask me anything ...请在这里打字问问题吧", 
                            on_change=clear_text,    
                            label_visibility='hidden')
    input_text = st.session_state["temp"]
    return input_text


    # Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.entity_memory.store = {}
    st.session_state.entity_memory.buffer.clear()

# Set up sidebar with various options
#with st.sidebar.expander("🛠️ ", expanded=False):
#    # Option to preview memory store
#    if st.checkbox("Preview memory store"):
#        with st.expander("Memory-Store", expanded=False):
#            st.session_state.entity_memory.store
#    # Option to preview memory buffer
#    if st.checkbox("Preview memory buffer"):
#        with st.expander("Bufffer-Store", expanded=False):
#            st.session_state.entity_memory.buffer
#    MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo','text-davinci-003','text-davinci-002','code-davinci-002'])
#    K = st.number_input(' (#)Summary of prompts to consider',min_value=3,max_value=1000)

MODEL = "gpt-3.5-turbo"
K = 100

with st.sidebar:
    st.markdown("---")
    st.markdown("# About")
    st.markdown(
       "ChatGPTm is ChatGPT added memory. "
       "It can do anything you asked and also remember you."
            )
    st.markdown(
       "This tool is a work in progress. "
            )
    st.markdown("---")
    st.markdown("# 简介")
    st.markdown(
       "ChatGPTm就是增加了记忆的ChatGPT。 "
       "你可以在右边的对话框问任何问题。"
            )
    st.markdown(
       "希望给国内没法注册使用ChatGPT的朋友带来方便！"
            )

    
# Set up the Streamlit app layout
st.title("🤖 ChatGPT with Memory 🧠")
#st.subheader(" Powered by 🦜 LangChain + OpenAI + Streamlit")

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

# Let user select version
st.write("GPT4.0上线了！无需注册就可以体验只有OpenAI付费用户才可以体验的GPT4.0了！")
version = st.selectbox("Choose ChatGPT version 请选择您想使用的ChatGPT版本", ("3.5", "4.0"))
if version == "3.5":
    # Use GPT-3.5 model
    MODEL = "gpt-3.5-turbo"
else:
    # USe GPT-4.0 model
    MODEL = "gpt-4"
    
# Ask the user to enter their OpenAI API key
#API_O = st.sidebar.text_input("API-KEY", type="password")
# Read API from Streamlit secrets
API_O = st.secrets["OPENAI_API_KEY"]

# Session state storage would be ideal
if API_O:
    # Create an OpenAI instance
    llm = OpenAI(temperature=0,
                openai_api_key=API_O, 
                model_name=MODEL, 
                verbose=False) 


    # Create a ConversationEntityMemory object if not already created
    if 'entity_memory' not in st.session_state:
            st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=K )
        
        # Create the ConversationChain object with the specified configuration
    Conversation = ConversationChain(
            llm=llm, 
            prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
            memory=st.session_state.entity_memory
        )  
else:
    st.sidebar.warning('API key required to try this app.The API key is not stored in any form.')
    # st.stop()


# Add a button to start a new chat
#st.sidebar.button("New Chat", on_click = new_chat, type='primary')

# Get the user input
user_input = get_text()

# Generate the output using the ConversationChain object and the user input, and add the input/output to the session
if user_input:
    if st.session_state["balance"] > -0.03:
        with get_openai_callback() as cb:
            output = Conversation.run(input=user_input)  
            st.session_state.past.append(user_input)  
            st.session_state.generated.append(output) 
            st.session_state["balance"] -= cb.total_cost * 4
    else:
        st.session_state.past.append(user_input)  
        if is_four_digit_number(user_input) :
            st.session_state["balance"] += st.session_state["deposit"]
            st.session_state.generated.append("谢谢支付，你可以继续使用了") 
        else: 
            st.session_state.generated.append("请用下面的支付码支付¥10后才可以再继续使用。我会再送你¥10元。支付时请记下转账单号的最后4位数字，在上面对话框输入这四位数字") 
        

# Allow to download as well
download_str = []
# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i],icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])
                            
    # Can throw error - requires fix
    download_str = '\n'.join(download_str)
    
    if download_str:
        st.download_button('Download 下载',download_str)

# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
        with st.sidebar.expander(label= f"Conversation-Session:{i}"):
            st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:   
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session
        
# Load the images
image1 = Image.open("wechatqrcode_leo.jpg")
image2 = Image.open("zhifubaoqrcode_kyle.jpg")
image3 = Image.open("paypalqrcode.png")
image4 = Image.open("drpang_shipinhao2.jpg")

# Display the image with text on top
st.write("I have to pay OpenAI API for each of your usage. Please consider donating $5 to keep this service alive! Thank you!")
st.write("您现在账上的余额是：", round (st.session_state["balance"]*7, 2), "人民币。")
st.write("我是史丹福机器人庞博士，我提供此应用的初衷是让国内的人也可以体验使用增加了记忆的ChatGPT。我在为你的每次使用支付调用OpenAI API的费用，包括3.5版，请扫码微信或支付宝支付¥10人民币来使用，我会再送你10元，按流量计费。")
st.write("因为没有你的注册信息，你如果关闭浏览器或这个网页，你的余额会归零，所以使用时尽量不要关闭浏览器或这个页面")
st.write("长期用户可交¥1688年费（和OpenAI付费用户收费一致），填上你的邮箱，我会发给你专属的小程序，记忆力是这个的10倍。")
st.write("OpenAI对GPT4.0 API的收费是3.5的20倍，请大家体验时注意。")
st.write("我在我的《史丹福机器人庞博士》微信视频号也有很多关于ChatGPT和怎样使用ChatGPT魔法的视频，还有怎么使用这个小程序的视频，欢迎白嫖。也有系统的课程《零基础精通掌握ChatGPT魔法6讲》和《ChatGPT和LLM应用编程7讲》给愿意知识付费的同学深入学习。 ")
st.write("所有已直播的课程在我的视频号主页的直播回放里， 每节课99元，第一节课大家可以免费试听。 如果想购买整套课程，有50%折扣，每个只要299元。可以在我的视频号主页私信我购买，注明ChatGPT魔法课或编程课。两门课都上再优惠100元，只要499元。")

#st.image(img, caption=None, width=200)

# Divide the app page into two columns
col1, col2, col3 = st.columns(3)

# Display the first image in the first column
with col1:
    st.image(image1, caption="微信支付", width=200)

# Display the second image in the second column
with col2:
    st.image(image2, caption="支付宝", width=200)

# Display the third image in the third column
with col3:
    st.image(image3, caption="PayPal", width=200)

st.image(image4, caption="史丹福机器人庞博士视频号，微信扫码前往", width=200)
