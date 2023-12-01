import streamlit as st

try:
    # Attempt to import langchain
    import openai
    print("openai is installed.")
    st.write("openai is installed.")
    print("Version:", openai.__version__)
except ImportError:
    print("openai is not installed.")
    st.write("openai is NOT installed.")

st.title("Hello world")
st.balloons()
