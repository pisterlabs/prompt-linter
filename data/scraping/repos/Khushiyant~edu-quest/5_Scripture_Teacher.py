import streamlit as st
import openai

hide_default_format = """
         <style>
            #MainMenu {visibility: hidden; }
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_default_format, unsafe_allow_html=True)


def generate_query(query):
    prompt = "Act as my religious scripture teacher of Gita, vedas, Kalki puran, Shiv Puran and don't answer anything which is not mentioned in them"

    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt}
        ],
        prompt=query,
        temperature=0.3,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    st.text(response.choices[0].text)


def main():
    st.header("Scripture Teacher")
    text_input = st.text_input(
        "Ask me anything about Gita, vedas, kalki puran, shiv puran")

    response = generate_query(f"{text_input}") if text_input else ""
    st.write(response)


if __name__ == '__main__':
    main()
