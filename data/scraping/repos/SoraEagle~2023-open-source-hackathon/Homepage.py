import time
import requests
from common import *
from PIL import Image
from io import BytesIO
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from githubqa.data_processing import dictionary_to_docs, create_retriever
from githubqa.get_info_from_api import get_avatar_info, get_repo_list, github_api_call
from streamlit_lottie import st_lottie



def load_lottieurl(url: str ):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

point_url = "https://lottie.host/5fbefebf-0144-4a35-a463-c9a2186c0fab/Jym0vP7Azn.json"
lottie_point = load_lottieurl(point_url)


# 1. Session Initialize & Donation
initialize_session()

st.set_page_config(
    page_title="Gitty"
)


# 2. Sidebar username input
st.sidebar.title("Gitty")
st.session_state["user_name"] = st.sidebar.text_input(
    "GitHub Username:",
    key="github_user_input",
    placeholder="input GitHub username",
    value=st.session_state["user_name"],
    on_change=handling_user_change,
)

# 3. Sidebar Select Repo + User Avatar layout
if st.session_state["user_name"]:
    user_name = st.session_state["user_name"]
    repo_list = get_repo_list(user_name)[0]
    user_info = get_avatar_info(user_name)
    if repo_list:
        repo_list = [DEFAULT_SELECT_VALUE] + repo_list
        st.session_state["repo_name"] = st.sidebar.selectbox(
            f"Select {user_name}'s repository",
            repo_list,
            key="repo_select",
            index=repo_list.index(st.session_state["repo_name"]),
        )
        if st.session_state["repo_name"] != DEFAULT_SELECT_VALUE:
            st.session_state[
                "repo_url"
            ] = f"https://github.com/{st.session_state['user_name']}/{st.session_state['repo_name']}"
        avatar_url = user_info["avatar_url"]
        image_response = requests.get(avatar_url)
        image = Image.open(BytesIO(image_response.content)).resize((250, 250))
        st.sidebar.image(
            image, use_column_width="always", caption=f"{user_name}'s profile"
        )
    else:
        st.error("Invalid username.")



# 4. Main Screen Start
st.header("Chatbot")
if st.session_state["repo_url"]:
    with st.spinner("Analyzing the repository..."):
        # Return Value : "File_name" : "File_content"
        github_info_dict, structure_content, _, user_content = github_api_call(
            st.session_state["repo_url"]
        )

    # Return Values [Doc1, Doc2 ...]
    with st.spinner("Embedding to Vectorspace..."):
        docs = dictionary_to_docs(
            github_info_dict,
            structure_content,
            user_content,
            chunking_size=1000,
            overlap_size=0,
            model_name=MODEL_NAME,
        )
        # Chunked Data to Vector embedding
        embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
        retriever = create_retriever(embedding_model, docs)

        if st.session_state["repo_url"] not in st.session_state["visitied_list"]:
            retriever = create_retriever(embedding_model, docs)
            st.session_state["retriever"] = retriever
            st.session_state["visitied_list"].append(st.session_state["repo_url"])
            st.session_state["messages"] = []
        else:
            retriever = st.session_state["retriever"]

    open_ai_model = ChatOpenAI(model_name=MODEL_NAME)

    if not st.session_state["chat_memory"]:
        st.session_state["chat_memory"] = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
    memory = st.session_state["chat_memory"]

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=open_ai_model,
        memory=memory,
        retriever=retriever,
        get_chat_history=lambda h: h,
        verbose=True,
    )
    # DEBUG
    # print("[DEBUG] Memory:", memory.load_memory_variables({}))

    # QA Start
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me about the repository!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("Generating an answer..."):
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                response = qa_chain({"question": prompt})  # QA chain
                for response in response["answer"]:
                    full_response += response
                    time.sleep(0.02)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
else:
    st.info(
        """
        Please input **GitHub Username** and **Repository Name** in the left sidebar.\n
        And I will answer questions for you about that repository!\n
        *Powered by Langchain, OpenAI, FAISS & Streamlit.*
        """
    )
    st_lottie(point_url, key="hello", height=500)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
           
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
