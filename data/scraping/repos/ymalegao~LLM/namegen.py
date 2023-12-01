import streamlit as st
import langchain_helper

import serpy

st.title("Food generator")

cuisine  = st.sidebar.selectbox("Pick a Cuisine", ("Indian", "Marathi", "Mexican", "Arabic", "American"))

location = st.sidebar.selectbox("Pick a City", ("Milpitas", "Santa Cruz", "Fremont"))

location += " , California"


if cuisine:
    s = f'Return the names of the best {cuisine} resturants in {location}'
    print(s)
    res = serpy.chooseandsearch(s)
    st.header("{} places near you".format(cuisine.capitalize()))
    menu_items = res
    st.write("*********")
    st.write("-",res)