import os
import openai
import streamlit as st
import requests
from time import perf_counter
from streamlit_pills import pills 


from components.sidebar import sidebar
from functions.writer import writer
from PIL import Image


image = Image.open("assets/favicon.png")
st.set_page_config(
    page_title="Khontenu",
    page_icon=image,
    layout="wide",
    menu_items={
        'Get help': 'mailto:eliott@khlinic.fr'
    }
)



st.header("✅ Khontenu pour les QCMs")
st.markdown("---")

if "shared" not in st.session_state:
   st.session_state["shared"] = True

sidebar()

openai.api_key = st.session_state.get("OPENAI_API_KEY")

st.markdown("#### v0.1 du prompt")
st.markdown("Modifications attendues : moins de notions non données dans le cours, meilleures corrections")

with st.expander("Contenu des annales", expanded=False):
    annale = st.text_area("Annales", placeholder="Une série de 6 à 10 QCMs d'annales")

col1, col2, col3 = st.columns(3)
submit = col3.button("Rédiger ✍🏻", use_container_width=1)
    
if submit:
    st.session_state["total_tokens"] = 0
    st.session_state["completion_tokens"] = 0
    st.session_state["prompt_tokens"] = 0
    st.session_state["error"] = 0

    with st.spinner("Requête en cours..."):
        ts_start = perf_counter()

        if st.session_state["error"] == 0:
            final_text = writer(annale)
            st.write(final_text)    
        
        ts_end = perf_counter()
        st.info(f" {round(((ts_end - ts_start)/60), 3)} minutes d'exécution")
        cost = st.session_state["prompt_tokens"] * 0.00003 + st.session_state["completion_tokens"] * 0.00006
        st.write("Coût de l'article : " + str(cost) + " $")
        col1, col2, col3 = st.columns([2, 2,1])
        rewrite = col3.button("Réécrire ✍🏻", use_container_width=1)

    col1, col2, col3 = st.columns([2, 2,1])
    col3.download_button(
        label="Télécharger 💾",
        data=final_text,
        file_name='qcm.txt',
        mime='text/plain',
    )
