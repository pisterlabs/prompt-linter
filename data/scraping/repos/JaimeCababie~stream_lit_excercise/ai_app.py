import streamlit as st
import re
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Initialize ChatOpenAI
llm = ChatOpenAI(model_name="gpt-4", temperature=0.5, openai_api_key="ADD_YOUR_API_KEY")

st.title("Textibot")

option = st.selectbox(
    "Selecciona un archivo:",
    ("Instructivo 1", "Instructivo 2", "Instructivo 3"),
    index=None,
    placeholder="Archivo",
)

initial_prompt = "Eres un exelente explicador de procesos, por favor responde preguntas sobre el instructivo proporcionado por el usuario. Si la pregunta no es sobre el intructivo, indicale al usuario, aun que los instructivos son ficticios, por favor responde con la mayor seriedad posible. Esto es una experiencia para el usuario y la intención es mantenerlo enganchado. Por favor no cambies de tema, y manten la conversación sobre el instructivo. "

def read_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        return f.read()

if option:
    if option == "Instructivo 1":
        option_info = read_file("instructivo_1.txt")
    elif option == "Instructivo 2":
        option_info = read_file("instructivo_2.txt")
    elif option == "Instructivo 3":
        option_info = read_file("instructivo_3.txt")
    
    if "show_instructivo" not in st.session_state:
        st.session_state["show_instructivo"] = False
    
    if st.button('Ver/Ocultar Instructivo'):
        st.session_state["show_instructivo"] = not st.session_state["show_instructivo"]

    if st.session_state["show_instructivo"]:
        st.markdown(option_info)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Preguntame sobre el " + option}]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if (prompt := st.chat_input()):
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        messages = [
        SystemMessage(content=initial_prompt),
        HumanMessage(content=option_info),
        AIMessage(content="preguntame sobre el " + option),
        HumanMessage(content=prompt),]
        llm_response =llm(messages).content
        st.chat_message("assistant").write(llm_response)
        st.session_state.messages.append({"role": "assistant", "content": llm_response})


