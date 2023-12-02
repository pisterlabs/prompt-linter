import os
import openai
import streamlit as st
from PIL import Image

openai_api_key = os.getenv("OPENAI_API_KEY")

image = Image.open('berend_gpt/images/chatachtergrond.png')
st.set_page_config(
        page_title=" : genie: Berend-Botje Skills",
        page_icon=" :genie: ",
        layout="wide",
        initial_sidebar_state="collapsed" )

col1, col2 = st.columns(2)

with col1:
        st.header(":genie: Berend-Botje Skills" )
        st.subheader(":male-teacher: De Basis - ChatGPT kloon - \n*waarom zou je moeilijk doen ....?* ")
        st.markdown(
                """ 
                ##### Dit is Berend's ChatBot. Een kloon van ChatGPT, en gebruikmakend van het slimme en snelle gpt-3.5-turbo model 
                
                ###### Dat betekent dat je vragen snel :comet: door Berend worden beantwoord, maar ook dat er beperkingen zijn zoals: 
                - **de kennis van Berend gaat tot :calendar: 2021, omdat het model getraind is met data tot aan 2021**
                - **dat niet alles wat Berend zegt de waarheid is. De huidige generatie AI modellen, kunnen, wanneer ze het antwoord niet weten, gewoon wat gaan verzinnen. 
                        Dit fenomeen noemen we :person_with_probing_cane: *hallucineren* en kan ook Berend gebeuren.** 
                - **Je werkt samen met Berend, waarbij Berend jouw assistent is, maar jij altijd degene moet zijn die bepaalt wat waar is of niet waar***
                """
        )
with col2:
        st.image(image, caption=None, width=240, use_column_width=True, clamp=True, channels="RGB", output_format="auto")
        st.markdown("""
                - **dat niet alles wat Berend zegt de waarheid is. De huidige generatie AI modellen, kunnen, wanneer ze het antwoord niet weten, gewoon wat gaan verzinnen. 
                        Dit fenomeen noemen we :person_with_probing_cane: *hallucineren* en kan ook Berend gebeuren.** 
                - **Je werkt samen met Berend, waarbij Berend jouw assistent is, maar jij altijd degene moet zijn die bepaalt wat waar is of niet waar***
                """)



# openai.api_key = st.secrets["OPENAI_API_KEY"]

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "system", "content": "Geef altijd antwoord in het Nederlands"})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] != "system":
            st.markdown(message["content"])

if prompt := st.chat_input("Hoe gaat het?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
