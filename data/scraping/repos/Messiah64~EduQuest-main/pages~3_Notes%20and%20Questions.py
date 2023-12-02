import openai
import streamlit as st

OpenAI_Key = st.secrets["OpenAI_Key"]

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

def generate_query(query):

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=query,
        temperature=0.3,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    response_text = response.choices[0].text
    st.text_area("Generated Response", value=response_text, height=200)


def main():

    st.header("Notes and Questions")
    text_input = st.text_input("Enter a title or topic")

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
