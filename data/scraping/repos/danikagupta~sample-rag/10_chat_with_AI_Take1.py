import openai
import streamlit as st

#
openai_key=st.text_input("OpenAI Key","")
openai.api_key = openai_key

#
user_question=st.text_input("Question","")
if st.button('Ask'):
    conversation = [{"role": "user", "content": user_question}]

    # Send a message to the assistant
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation
    )
    # Print the assistant's reply
    st.write(response.choices[0]['message']['content'])
    with st.expander("See full JSON"):
        st.write(response)