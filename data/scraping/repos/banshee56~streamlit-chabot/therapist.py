import openai
import streamlit as st

def generate_response(myprompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Your role is to act like a licenced therapist. Give your answers accordingly###" + myprompt,
        temperature=.3,                 # recommended: 0.3 to 0.7
        max_tokens=1024
    )
    # print (response.choices)
    return response.choices[0].text.strip()

def main ():
    st.title("Friendly Chat Therapist")
    myprompt = st.text_input("I'm Dr Fraiser Crane. I'm listening, how can I help you?\n")
    if st.button("Submit"):
        st.write(generate_response(myprompt))

if __name__ == "__main__":
    main()