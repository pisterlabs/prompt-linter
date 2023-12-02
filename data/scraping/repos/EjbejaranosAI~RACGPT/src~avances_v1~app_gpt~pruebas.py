import openai
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

openai.api_key = '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'

st.markdown("<h1 style='text-align: center; color:white; font-size: 50px;'>RACGPT</h1>",
            unsafe_allow_html=True)

options = option_menu(None, ["Info", "RacGpt", "ChatBot", 'Settings'],
                        icons=['book', 'cloud-upload', "list-task", 'gear'],
                        orientation="horizontal",
                        styles={
                            "container": {"padding": "opx", "background-color": "#0a4d75"},   # opciones de la barra
                            "icon": {"color": "white", "font-size": "20px"},                        # Color and Size icons
                            "nav-link": {"font-size": "20px", "text-align": "center", "margin": "1px",
                                         "--hover-color": "#eee"},
                            "nav-link-selected": {"background-color": "#126fa8"},
                        })

if options == "Info":
    img = Image.open(r'C:\Users\dstra\Desktop\app_gpt\libro.png')
    st.markdown("<h1 style='text-align: center; color:white; font-size: 50px;'>Welcome to RACGPT</h1>", unsafe_allow_html=True)
    st.write("")

    col1, col2, col3 = st.columns([0.1, 0.1, 0.1])
    col2.image(img, use_column_width=True)

if options == "RacGpt":
    st.write("""
    
     # .-.
    
    """)


if options == "ChatBot":
    def chatbot_app(message):
        # Make a request to the ChatGPT model
        response = openai.Completion.create(
            engine="text-davinci-003",  # Choose the engine that suits your requirements
            prompt=message,
            max_tokens=100,  # Adjust the response length as needed
            temperature=0.4,  # Adjust the temperature to control randomness
            n=1,  # Generate a single response
            stop=None,  # Add custom stop conditions if needed
            timeout=10,  # Adjust the timeout as needed
        )
        # Extract the generated response
        reply = response.choices[0].text.strip()
        return reply

    user_input = st.text_input("Ask me something")

    if user_input:
        # Pass the user input to the chatbot function
        bot_reply = chatbot_app(user_input)
        st.text_area("ChatGPT", value=bot_reply, height=200)





# streamlit run pruebas.py