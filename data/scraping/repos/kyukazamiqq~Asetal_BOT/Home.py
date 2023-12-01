import streamlit as st
import streamlit_authenticator as stauth
from openai import OpenAI
import yaml
from yaml.loader import SafeLoader
from st_pages import Page, show_pages, add_page_title, hide_pages


with open('.streamlit/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

st.set_page_config(page_title="Asetal - Asisten Kesehatan Mental", page_icon=":robot:")

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status:
    hide_pages(["Register"])
    st.sidebar.title(f"Halo! Selamat datang {name}")

    # Clear chat messages when a new user logs in
    st.session_state.messages = []

    # Inisiasi Halaman
    if 'active_page' not in st.session_state:
        st.session_state.active_page = 'Home'
        st.session_state.radiobuttons = 'Home' 

    def home():
        st.header("Home!")
        st.markdown("Hai manusia kuat! aku tau ini berat buat kamu namun kamu hebat bisa sampai di titik ini. Terus lanjutkan perjuangan kamu untuk dapat menikmati indahnya duniawi, kamu boleh saja lelah tapi jangan berhenti")

    def chat_with_terapis():
        st.header("Konsultasi dengan teman sekitar")

        # Define the chat file path
        chat_file_path = "chat_history.yaml"

        # Check if terapis_messages is not in st.session_state
        if "terapis_messages" not in st.session_state:
            st.session_state.terapis_messages = []

            # Load existing chat messages from YAML file
            with open(chat_file_path, "r") as file:
                st.session_state.terapis_messages = yaml.safe_load(file) or []

        # Display existing chat messages
        for message in st.session_state.terapis_messages:
            st.text(f"{message['sender']}: {message['content']}")

        # Get user input
        user_input = st.text_input("Kirim pesan anda ke terapis")

        # Send user input when button is clicked
        send_button = st.button("Kirim Pesan")
        if send_button and user_input:
            st.session_state.terapis_messages.append({"sender": name, "content": user_input})
            st.text(f"{name} : {user_input}")

            # Save updated chat history to YAML file
            with open(chat_file_path, "w") as file:
                yaml.dump(st.session_state.terapis_messages, file)

    def chat_with_ai():
        st.header("Asetal Bot!")
        st.markdown("Here to help you out of your mental health problem !")

        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-3.5-turbo"

        if "messages" not in st.session_state:
            st.session_state.messages = []

        with st.chat_message("assistant"):
            st.markdown("Halo nama saya Asetal Bot, Asisten Kesehatan Mental Kamu, Tolong ceritakan masalah kamu")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Gimana kabar kamu hari ini?"):
            # Ensure the user input is focused on emotions and mental well-being
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                for response in client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                ):
                    full_response += (response.choices[0].delta.content or "")
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

    def CB_RadioButton():
        st.session_state.active_page = st.session_state.radiobuttons

    if st.session_state.active_page == 'Home':
        home()
    elif st.session_state.active_page == 'Asetal Bot':
        chat_with_ai()
    elif st.session_state.active_page == 'Konsultasi dengan Ahlinya':
        chat_with_terapis()

    st.sidebar.radio('Menu', ['Home', 'Asetal Bot', 'Konsultasi dengan Ahlinya'], key='radiobuttons', on_change=CB_RadioButton)
    authenticator.logout("Logout", "sidebar")
