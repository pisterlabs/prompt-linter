import langchain_helper as lch
import streamlit as st

st.title("pet name generator")

animal_type = st.sidebar.selectbox("what is your pet?", ("cat", "dog", "cow"))

if animal_type == "cat":
    pet_color = st.sidebar.text_area(label="what is your pet color?", max_chars=15)
elif animal_type == "dog": 
    pet_color = st.sidebar.text_area(label="what is your pet color?", max_chars=15)
else:
    pet_color = st.sidebar.text_area(label="what is your pet color?", max_chars=15)

if pet_color:
    response = lch.generate_pet_name(animal_type, pet_color)
    st.text(response)