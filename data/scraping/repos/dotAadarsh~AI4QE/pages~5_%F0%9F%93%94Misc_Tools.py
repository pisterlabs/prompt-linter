import openai
import streamlit as st

openai.api_key = st.secrets["OPENAI_KEY"]

def generate_query(query):

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=query,
        temperature=0.3,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    st.text(response.choices[0].text)


def main():

    st.header("Misc tools")
    text_input = st.text_input("Enter the title or topic")

    if text_input:
        option = st.selectbox(
            'What are the actions you want to perform?',
            ['Notes', 'Questions'])

        if option == 'Notes':
            response = generate_query(f"Generate notes on the topic: {text_input}")
            st.write(response)
        elif option == 'Questions':
            response = generate_query(f"Generate list of questions on the topic: {text_input}")
            st.write(response)

if __name__ == '__main__':
    main()
