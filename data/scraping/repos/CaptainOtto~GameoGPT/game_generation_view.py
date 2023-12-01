from models import models
from openai_generation import generate_result 

import streamlit as st
from streamlit_chat import message

def game_generation_view():
    st.title("Game generation")

    engineOption = st.selectbox("Engine", ["Unreal Engine", "Unity", "Godot"])
    dimensionOption = st.selectbox("Dimensions", ["2D", "3D"])
    artStyleOption = st.selectbox("Art Style", ["Realistic", "Stylized Realism", "Stylized", "Pixel art", "Low Poly"])
    genreOption = st.multiselect("Genre", ["Survival", "Fighting", "Racing", "Puzzle", "Action", "Fps", "Strategy", "Rpg", "Stealth"])

    optionalDescription = st.text_area("Description", "")

    models.game_gen_models.engine = engineOption
    models.game_gen_models.dimension = dimensionOption
    models.game_gen_models.artStyle = artStyleOption
    models.game_gen_models.genre = genreOption

    if optionalDescription:
        models.game_gen_models.description = optionalDescription

    if st.button("Generate"):
        st.write("---")

        st.spinner("Generating...")

        with st.spinner("Generating..."):
            generation_result = generate_result()

            if generation_result == None:
                return

            message(generation_result)

            st.download_button("Download", generation_result)