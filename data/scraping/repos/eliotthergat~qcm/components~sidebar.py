import os
import openai
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def sidebar():
    with st.sidebar:
        st.markdown(
                "## Comment fonctionne Khontenu ?\n"
                "1. 🔑 Entrez une clé OpenAI \n"
                "2. 🏴‍☠️ Choisissez les annales sourcces \n"
                "3. 🖊️ Lancez la rédaction \n"
            )
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Collez votre clé OpenAI ici",
            help="Nécessaire pour utiliser l'API",
            value=os.environ.get("OPENAI_API_KEY", None)
            or st.session_state.get("OPENAI_API_KEY", ""),
        )
        st.session_state["OPENAI_API_KEY"] = api_key_input
        st.markdown("---")
        st.markdown("# Paramètres")
        max_tokens = st.slider("Longueur maximale (`max_tokens`):", min_value=1, max_value=8000, value=st.session_state.get("MAX_TOKENS", 4000), step=25, help="Nombre maximum de tokens à utiliser")
        st.session_state["MAX_TOKENS"] = max_tokens
        
        st.markdown("## (Ne pas toucher)")
        temperature = st.slider("Température (`randomness`):", min_value=0.0, max_value=2.0, value=st.session_state.get("TEMPERATURE", 1.0), step=0.1, help="###")
        st.session_state["TEMPERATURE"] = temperature

        presence_penalty = st.slider("Pénalité de présence (`presence_penalty`):", min_value=0.0, max_value=2.0, value=st.session_state.get("PRESENCE_PENALTY", 0.0), step=0.01, help="###")
        st.session_state["PRESENCE_PENALTY"] = presence_penalty

        frequency_penalty = st.slider("Pénalité de fréquence (`frequency_penalty`):", min_value=0.0, max_value=2.0, value=st.session_state.get("FREQUENCY_PENALTY", 0.0), step=0.01, help="###")
        st.session_state["FREQUENCY_PENALTY"] = frequency_penalty

        max_retries = st.slider("Nombre d'essais (`max_retries`):", min_value=1, max_value=5, value=st.session_state.get("max_retries", 3), step=1, help="Nombre de tentatives en cas d'erreur de l'API")
        st.session_state["max_retries"] = max_retries

        wait_time = st.slider("Temps d'attente (`wait_time`):", min_value=1, max_value=20, value=st.session_state.get("wait_time", 5), step=1, help="Attente en secondes avant un nouvel appel API")
        st.session_state["wait_time"] = wait_time

        st.markdown("---")
        st.markdown("# À propos")
        url = "https://khlinic.fr"
        st.markdown(
            "📖 Tous les crédits appartiennent à [Khlinic](%s)." % url
        )
        hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)
        
