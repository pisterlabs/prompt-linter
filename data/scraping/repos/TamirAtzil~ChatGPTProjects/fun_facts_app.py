import openai
import streamlit as st

def get_fun_fact(topic):
    openai.api_key = st.secrets["MY_OPENAI_API_KEY"]

    prompt_text = f"Tell me a fun fact about {topic}."

    response = openai.Completion.create(
      model="text-davinci-002",
      prompt=prompt_text,
      max_tokens=150
    )

    fact = response.choices[0].text.strip()

    return fact

# Streamlit UI
st.set_page_config(page_title="Fun Fact Generator By The LEGENDARY Farsi 👳🏾‍", layout="centered")

st.markdown("""
    <style>
        .reportview-container {
            background-color: #f0f0f0;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Fun Fact Generator By The LEGENDARY Farsi 👳🏾")
st.markdown("""
Come get some knowledge and wisdom from the Sexiest Farsi exists, to become better version of yourself 🧠! 
""")

topic = st.text_input("🔍 Enter a topic that you want the Farsi to tell you about:")

if topic:
    with st.spinner(f"The Farsi knows everything about {topic}..."):
        fact = get_fun_fact(topic)
    st.success(fact)

st.markdown("""
---
**Every day is a school day!** 📘
""")
