import os
import openai
import streamlit as st
from PIL import Image

try:
    openai_api_key = os.getenv("OPENAI_API_KEY")
except:
    openai_api_key = st.secrets["OPENAI_API_KEY"]



image = Image.open('berend_gpt/images/chatting.png')
st.set_page_config(
        page_title=" :genie: Berend Skills",
        page_icon=" :genie: ",
        layout="wide",
        initial_sidebar_state="collapsed" )

col1, col2 = st.columns(2)

with col1:
        st.header(":genie: Berend Skills" )
        st.subheader(":male-teacher: De Rollenspeler\n*waarom zou je moeilijk doen ....?* ")
        st.markdown(
                """ 
                ##### Dit is Berend's Rollenspeler. De Rollenspeler kan helpen bij het oefenen van bepaalde vaardigheden door middel van een rollenspel. Jij geeft aan welke rol Berend speelt, en welke rol jij hebt. 
                
                ###### Jij vraagt: "Speel een rollenspel, waarbij jij de rol van een klant speelt, en ik de rol van een stagaire die bij een kinderdagverblijf werkt." Berend zal dan het rollenspel starten aan de hand van een Case. **Belangrijk: wanneer je wilt stoppen type je: "STOP SPEL", waarna je van Berend feedback krijgt over jouw rol bij deze case.**  
                """
        )
with col2:
        st.image(image, caption=None, width=240, use_column_width=True, clamp=True, channels="RGB", output_format="auto")
        st.markdown(""" """)



# openai.api_key = st.secrets["OPENAI_API_KEY"]

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {
        "role": "system", 
        "content": 
        """
        We spelen een rollenspel, jij ( assistant ),  speelt de rol van Klant, en 
        de gebruiker ( user ) speelt de rol van Stagaire die bij een bepaald bedrijf werkt. 
        Bij dit rollenspel verloopt een conversatie om en om. Jij ( assistant ) bent Klant en stelt een vraag, en 
        wacht dan op het antwoord van de Stagaire ( user ). Daarna geef jij, de Klant, weer antwoord. Enzovoort.
        Het rollenspel verloopt dus stap voor stap: 
        1. Eerst vraagt de gebruiker om een rollenspel te starten en geeft daarbij aan: 
            - wat jouw rol als Klant is, 
            - wat de User rol als Stagaire is, 
            - in welke setting het afspeelt
        2. Op basis van vraag start jij, als Klant, een interessante case: {Verzonnen_Case}, 
        door een vraag te stellen aan de Stagaire gebaseerd op je {Verzonnen_Case}. Bijvoorbeeld: 'Klant: Goedendag, mijn naam is Berend. en dan de vraag uit { Verzonnen_Case }'.
        Dan wacht je op het antwoord van de Stagaire, voordat jij zelf weer een antwoord geeft. 
        Het gesprek is afgelopen als de Klant tevreden is, of als de stagaire dit expliciet aangeeft met: "STOP SPEL".
        3. Nadat het rollenspelt is gestopt, geef jij Feedback op het handelen van de Stagaire. 
        De bedoeling van het rollenspel is dat de stagaire hiervan leert zodat hij/zij toekomstige soortgelijke werkelijke cases juist afhandeld. Geef altijd antwoord in het Nederlands
        """
        } 
    )
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
