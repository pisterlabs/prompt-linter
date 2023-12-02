import os
import openai
import streamlit as st
from PIL import Image
import pandas as pd

MIJNDATA = "berend_gpt/pages/opleidingen.csv"
df = pd.read_csv(MIJNDATA)
keuze = set(list(df["OPLEIDINGEN"]))
keuze = sorted(keuze)


try:
    openai_api_key = os.getenv("OPENAI_API_KEY")
except:
    openai_api_key = st.secrets["OPENAI_API_KEY"]



image = Image.open('berend_gpt/images/chatachtergrond.png')

st.set_page_config(
        page_title=":genie: Berend Skills",
        page_icon=":genie:",
        layout="wide",
        initial_sidebar_state="collapsed" )

col1, col2 = st.columns(2)

with col1:
        st.header(":genie: Berend Skills" )
        st.subheader(":male-teacher: De Rollenspeler -\n*waarom zou je moeilijk doen ....?* ")
        st.markdown(
                """ 
                ##### Dit is Berend's Rollenspeler. De Rollenspeler kan helpen bij het oefenen van gespreksvaardigheden 
                door middel van een rollenspel. 
                
                ###### Kies je opleiding, en de case die je wilt oefenen. Type dan **start** gevolgd door **Enter** 
                de simulatie wordt gestart. 
                Om te stoppen voer je **STOP SPEL** in, en vervolgens krijg je van Berend feedback over je prestatie
                """
        )
with col2:
        st.image(image, caption=None, width=240, use_column_width=True, clamp=True, channels="RGB", output_format="auto")
        st.markdown(""" """)


try:
        case = st.radio(
        "Waarmee wil je oefenen? 👇",
        ["sollicatiegesprek", "gesprek met klant", "voortgangsgesprek met mentor", "voortgangsgesprek met leidingevende"],
        key="case",
        #label_visibility=st.session_state.visibility,
        #disabled=st.session_state.disabled,
        #horizontal=st.session_state.horizontal,
        )
        opleiding = st.multiselect("JOUW OPLEIDING", keuze, [])
        
        
        if not opleiding:
            st.error("Selecteer een Opleiding")
            st.stop()
        else:
            print("GEKOZEN")
            print(opleiding[0])
            print(case)
            
            if case == "sollicatiegesprek":
                rol_berend = "leidingevende waarmee de student een sollicatiegesprek voert voor de functie " + opleiding[0]
                rol_user = "student die solliciteert naar de functie van " + opleiding[0]
            elif case == "gesprek met klant":
                rol_berend = "klant van " + opleiding[0]
                rol_user = "stagaire die werkt als  " + opleiding[0] 
            elif case == "voortgangsgesprek met mentor":
                rol_berend = "mentor van de student op school"
                rol_user = "student " + opleiding[0] + " die een voortgangsgesprek met de mentor voert"
            elif case == "voortgangsgesprek met leidingevende":
                rol_berend = "leidinggevende van de student"
                rol_user = "student die werkzaam is als " + opleiding[0] + " en die een voortgangsgesprek met de leidingevende voert"
               
            prompt = "Speel een rollenspel waarbij jij (Berend) de rol speelt van " + rol_berend  + ", en de gebruiker ( user ) de rol speelt van " + rol_user 
            
            

except:
    st.write("FOUT")
    st.stop()

if not prompt:
    st.stop()
# openai.api_key = st.secrets["OPENAI_API_KEY"]

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "system", "content": """Jij gaat een rollenspel spelen, waarbij jij de rol van Klant aanneemt, en de gebruiker de rol van Stagair die bij een bepaald bedrijf werkt. Als de gebruiker je gevraagd heeft om het rollenspel te starten, begin jij het rollenspel als Klant en stel je een vraag op basis van eendoor jou verzonnen case, die past bij de de gegevens die de gebruiker in de vraag aan je heeft verstrekt. Je wacht dan op het antwoord van de Stagair. Je wacht dus na jouw antwoord/vraag altijd op respons van de stagiare. Geef altijd antwoord in het Nederlands"""})
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] != "system":
            st.markdown(message["content"])



if prompt:
    print(prompt)

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
