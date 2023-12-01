import langchain_helper as lch
import streamlit as st

st.title('Lanchain Practice')

animal_type = st.sidebar.selectbox("Whay is your pet?", ["dog", "cat", "cow"])

if animal_type == "dog":
    pet_color = st.sidebar.text_area(
        "What is the color of your dog?", max_chars=15)
elif animal_type == "cat":
    pet_color = st.sidebar.text_area(
        "What is the color of your cat?", max_chars=15)
elif animal_type == "cow":
    pet_color = st.sidebar.text_area(
        "What is the color of your cow?", max_chars=15)

if pet_color:
    response = lch.generate_pet_name(animal_type, pet_color)

    st.text(response['pet_name'])
