# https://github.com/alejandro-ao/langchain-chat-gui was used as a starting point
# pip install streamlit streamlit-chat langchain openai
# Must use Python 3.7 or greater bc assumes dictionaries maintain their order
import streamlit as st
from streamlit_chat import message
import os
import openai
from datetime import datetime as dt
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from ChatDatabase import *

# Init temp vars
stss = st.session_state
chatModified = False
DEBUG = False


def NewChat():
    st.write("NewChat fn is running")
    stss.newChatSwitch = True
    if "messages" in st.session_state:
        del stss["messages"]  # clear old chat messages if any
    st.session_state.messages = [SystemMessage(content="You are a helpful assistant.")]
    stss.chatID = str(dt.now())
    stss.chatDict = {}


def NewUserSim():
    if DEBUG:
        st.write("NewUserSim fn is running")
    delete_user_state(stss.userID)
    delete_all_chats()
    if DEBUG:
        st.write("Values currently in session_state:")
    for k in st.session_state:
        del st.session_state[k]
    if DEBUG:
        st.write("After deleting state, this is what state looks like: ")
    for k in st.session_state:
        st.write("     " + str(k) + "  = " + str(st.session_state[k]))


def NewSessionSim():
    if DEBUG:
        st.write("NewSession fn is running")
    del st.session_state["messages"]


def initSessVars():
    if DEBUG:
        st.write("initSessVars is running")
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    # st.session_state.messages = [SystemMessage(content="You are a helpful assistant.")]
    stss.chatDict = {}
    stss.titlesList = []


def tempSliderChange():
    update_session_state_by_user(stss.userID, "Temperature", stss.tempSlider)


def buildChatDict(messages):  # TODO Rename to APPEND
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            stss.chatDict.update({str(i) + "_user": msg.content})
        else:
            stss.chatDict.update({str(i) + "_ai": msg.content})
    if DEBUG:
        st.write(stss.chatDict)
    # return chatDict


# only called when messages does not exist and chatDict has been retrieved
def ExtractChatData(RetrievedChat):
    stss.messages = []
    stss.messages.append(
        SystemMessage(content="You are a helpful assistant.", additional_kwargs={})
    )

    for i, (k, v) in enumerate(RetrievedChat["Content"].items()):
        if i % 2 == 0:
            # print("HumanMessage(content= " + v + ")")
            stss.messages.append(HumanMessage(content=v))
        else:
            # print("AIMessage(content= " + v + ")")
            stss.messages.append(AIMessage(content=v))

    stss.chatTitle = RetrievedChat["ChatTitle"]
    stss.chatID = RetrievedChat["ChatID"]


# NewUserSim()  # TODO:  Take out

# Check for valid user id at new session start
if "messages" not in st.session_state:  # new session
    NewSession = True
    try:
        stss.userID = st.secrets["USER_ID"]
    except FileNotFoundError:
        st.write("No SECRETS.TOML file found")
    except KeyError:
        st.write("No key named USER_ID found in Secrets file.")
    else:
        stss.userID = st.secrets["USER_ID"]
else:
    NewSession = False


# Test for first time user and init if so
if getUserState(stss.userID) == None:  # New user
    if DEBUG:
        st.write("First-time user detected and is being set up.")
    # set up sess vars first so that 'messages' has the system message
    initSessVars()
    # init and persist db persisted vars
    stss["tempSlider"] = 0.0  # initial value of temp slider
    save_newUserState(stss.userID, stss.tempSlider)
    # chatDict, chatTitle, chatID are not persisted till first chat created
    NewChat()
else:
    if NewSession:  # new session/startup
        if DEBUG:
            st.write("Not new user, but new session detected.")

        # Retrieve persisted state variables
        getResult = getUserState(stss.userID)  # pull in state
        stss.tempSlider = getResult["Temperature"]
        initSessVars()  # stss.chatDict is init'd here

        # Retrieve chat vars for this user, or init if necessary
        docsList = get_all_titles(stss.userID)  # returns list of dicts
        if len(docsList) > 0:  # there are one or more chats persisted for this userid
            for doc in docsList:
                stss.titlesList.append(doc["ChatTitle"])
            RetrievedChat = get_latest_ChatRecord(stss.userID)
            # TODO Extract messages from chatDict
            ExtractChatData(RetrievedChat)
            stss.newChatSwitch = False
        else:  # no chats to retrieve, so init state vars as needed
            NewChat()


with st.sidebar:
    st.title("Sid's ChatGPT Clone")

    btnNewChatReturn = st.button("New Chat", on_click=NewChat)
    # btnNewUserReturn = st.button("New User", on_click=NewUserSim)
    # btnNewUserSession = st.button("New Session", on_click=NewSessionSim)

    # titlesList = ["Title1", "Title2", "Title3", "Title4", "Title5"]
    for title in stss.titlesList:
        title

    # TODO Put message hx here, individually selectable, and in historical order

    # borrowed from https://github.com/dataprofessor/llama2
    temp = 0.0
    temp = st.sidebar.slider(
        "Temperature",
        min_value=0.01,
        max_value=5.0,
        step=0.01,
        on_change=tempSliderChange,
        key="tempSlider",
    )

    # top_p = st.sidebar.slider(
    #     "top_p", min_value=0.01, max_value=1.0, value=0.9, step=0.01
    # )
    # max_length = st.sidebar.slider(
    #     "max_length", min_value=64, max_value=4096, value=512, step=8
    # )

chat = ChatOpenAI(temperature=stss.tempSlider)

if DEBUG:
    st.write("Before prompt is submitted, state looks like this:")
    for k in st.session_state:
        st.write("------  " + str(k) + "  = " + str(st.session_state[k]))

# User-provided prompt
if prompt := st.chat_input("Enter your message: "):  # string
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.spinner("Thinking..."):
        response = chat(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=response.content))
        CurrContent = response.content
        chatModified = True


# display message history
if DEBUG:
    st.write("About to write messages out to screen.")
messages = st.session_state.get("messages", [])  # this is a list

for i, msg in enumerate(messages[1:]):
    if i % 2 == 0:
        message(msg.content, is_user=True, key=str(i) + "_user")
    else:
        message(msg.content, is_user=False, key=str(i) + "_ai")


if DEBUG:
    st.write(
        "newChatSwitch = "
        + str(stss.newChatSwitch)
        + ". -----len(Messages) = "
        + str(len(messages))
    )

# Create a new chat with new title.  Occurs after first chat turn so that title can be created.
if (
    stss.newChatSwitch == True and len(messages) > 1
):  # it's a new chat and one chat turn has occurred
    if DEBUG:
        st.write("newChatSwitchcode is running")
    chatTitle = chat(
        messages[1:]
        + [
            HumanMessage(
                content="What is a good title for this chat that is 20 characters or less?"
            )
        ]
    )
    stss.chatTitle = chatTitle.content
    stss.titlesList.append(stss.chatTitle)

    buildChatDict(messages)

    newChatSaveResult = save_new_chat(
        stss.userID, stss.chatID, stss.chatTitle, stss.chatDict
    )
    stss.newChatSwitch = False
    chatModified = False
    st.experimental_rerun()  # force sidebar to update with new chat title

if DEBUG:
    st.write("At end of run, state looks like this:")
    for k in st.session_state:
        st.write("------  " + str(k) + "  = " + str(st.session_state[k]))

if chatModified:
    buildChatDict(messages)

    # save to db
    upsertResult = upsertChatContent(stss.chatID, stss.chatDict)
    if DEBUG:
        st.write("The id of the record upserted is: " + str(upsertResult.upserted_id))
        st.write("Number of records modified = " + str(upsertResult.modified_count))
    # chatModified = False
    # NewSession = False


# Checklist
#    make new session work correctly
#        retrieve state vars, Chat Titles, and latest chat messages
#    create chat title list on sidebar

# TODO's:
# all sessions are loaded from db on startup?????


# Example:
# messages =
# [SystemMessage(content='You are a helpful assistant.', additional_kwargs={}),
# HumanMessage(content='What was the cause of the death of James Dean?', additional_kwargs={}, example=False),
# AIMessage(content='James Dean died in a car accident on September 30, 1955. The cause of the accident was attributed to speeding. He was driving his Porsche 550 Spyder when he collided with another car at an intersection near Cholame, California.', additional_kwargs={}, example=False)]

# chatDict = {'_id': ObjectId('650a0f31ebb39d26b56ba930'),
#  'UserID': 'sidjnsn66',
#  'ChatID': '2023-09-19 16:13:51.717162',
#  'ChatTitle': '"Fission Discovery"',
#  'Content': {'0_user': 'Who was the first person to discover the fissibility of materials?',
#              '1_ai': 'The discovery of the fissibility of materials, specifically the process
#                       of nuclear fission, is attributed to Otto Hahn and Fritz Strassmann.
#                       In 1938, they conducted experiments that led to the identification of nuclear
#                       fission, which involves the splitting of atomic nuclei. This groundbreaking
#                       discovery laid the foundation for the development of nuclear energy and atomic
#                       weapons.'
#             }
#            }
