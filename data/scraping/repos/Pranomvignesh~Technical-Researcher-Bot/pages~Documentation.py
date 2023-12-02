import streamlit as st


st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Technical Researcher Bot - Documentation",
    page_icon="🤖",
    menu_items={
        'Report a bug': "https://github.com/Pranomvignesh/Technical-Researcher-Bot/issues"
    }
)
st.title("Documentation")

st.markdown("""
**Author** - Vignesh Prakash - [LinkedIn](https://www.linkedin.com/in/pranomvignesh/)

**This is a technical preview of the Technical Researcher bot**

- This application works by fetching technical papers from the topic provided using Metaphor's search API
- The PDF of the technical paper is fetched and extracted
- This extracted content forms the memory of the chatbot
- ConversationalRetrievalChain from LangChain is used to build the bot



## Upcoming Improvements
1. Adding the reference link of the paper from which the answer is retrieved.
    - This can be achieved by adding custom system prompts with sample templates during retrieval
2. Creating a Chrome Extension, where research papers can be added under different folders
    - This will serve as a chatbot and a note taking tool during active research work
3. Creating Login Profiles, thereby we can save our chatbot instance under respective user id
""")

