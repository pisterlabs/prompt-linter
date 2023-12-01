import streamlit as st
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from st_pages import Page, show_pages, hide_pages, add_page_title
from streamlit_extras.mention import mention
import numpy as np
import openai
import requests
import json

end_point_chat = "http://ec2-18-191-126-188.us-east-2.compute.amazonaws.com:5000/chat"


add_page_title("Anahuac Copilot", page_icon="🤖")
show_pages(
    [
        Page("cliente.py", "Mentores", "👤"),
        Page("other_pages/️empresa.py", "Admin", ":gear:"),
    ]
)
hide_pages("Admin")

with st.sidebar:
    st.image("resources/Logo.png", use_column_width=True)
    st.title('Bienvenido a Mayab Copilot')
    st.write(
        "Soy una inteligencia artificial para ayudar a los mentores a responder preguntas de los alumnos. Estoy entrenado con información de:")
    st.write("- Reglamento Anáhuac Mayab 📖")
    st.write("- Directorio Telefónico 📞")
    st.write("- Tramites y servicios 📝")
    st.write("- Errores en plataformas 🤖")
    st.sidebar.write("")
    add_vertical_space(5)

# Generate a random ID for the user
if 'id' not in st.session_state:
    st.session_state['id'] = np.random.randint(1, 214483647)
    # put limit to 214483647, bigger number would be int 64 and that gave problems when serializing to json

id = st.session_state['id']
chat = {"id_usuario": id}


if 'generated' not in st.session_state:
    st.session_state['generated'] = [
        "¡Hola! Soy Anahuac Copilot 🤖, tu asistente para dudas académicas y más 📚✨. Aunque estoy en fase de prueba 🚧, ¡estoy aquí para apoyarte! 😊 No compartas info personal 🚫. ¿En qué puedo ayudarte hoy? 🌟"]

if 'past' not in st.session_state:
    st.session_state['past'] = ['¿Quién eres? 😯']

colored_header(label='', description='', color_name='orange-80')
response_container = st.container()
input_container = st.container()


def get_text():
    input_text = st.text_input("Escribe tu pregunta: ", "", key="input")
    return input_text


with input_container:
    user_input = get_text()

with response_container:
    if user_input:
        st.session_state.past.append(user_input)
        chat = {"id_usuario": id, "message": user_input}
        resp2 = requests.post(end_point_chat, json=chat)
        print(resp2.text)
        json_object = json.loads(resp2.text)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(json_object['response'])

    if st.session_state['generated']:
        fi = len(st.session_state['generated']) - 1
        for i in range(len(st.session_state['generated'])):
            with st.chat_message(name="user"):
                st.write(st.session_state['past'][i])
            with st.chat_message(name="assistant", avatar="resources/AI.png"):
                st.write(st.session_state['generated'][i])

mention(
    label="Desarrollado por VHuman.ai",
    url="https://Vhuman.ai",
)
