import streamlit as st
import openai
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import json

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True
    
if check_password():
    @st.cache_data()
    def fetch_content_from_link(link):
        response = requests.get(link)
        soup = BeautifulSoup(response.content, "html.parser")
        text = ' '.join([p.text for p in soup.find_all('p')])
        return text

    def get_input():
        input_type = st.selectbox("Select input type", ["Text","Link", "File"])
        temperature = st.slider("Creativity(0% - 100%):", value=0.5, min_value=0.0, max_value=1.0, step=0.1)
        st.write("Copy the following from the Example text")
        cols = st.columns(2)
        tone = cols[0].checkbox("Tone")
        voice = cols[0].checkbox("Voice")
        vocabulary = cols[1].checkbox("Vocabulary")
        sentence_structure = cols[1].checkbox("Sentence structure")
        options = [option for option, checked in {"tone": tone, "voice": voice, "vocabulary": vocabulary, "sentence structure": sentence_structure}.items() if checked]
        if input_type == 'Text':
            return st.text_area("Enter the Example text:"), input_type, temperature, options
        # elif input_type == "File":
        #     uploaded_file = st.file_uploader("Upload a file:")
        #     if uploaded_file:
        #         text = uploaded_file.read().decode('utf-8')
        #         return text, input_type, temperature, options
        #     else:
        #         return None, input_type, temperature, options
        elif input_type == "Link":
            link = st.text_input("Enter the link:")
            if link:
                text = fetch_content_from_link(link)
                print(text)
                return text, input_type, temperature, options
            else:
                return None, input_type, temperature, options
            

    def get_prompt():
        return st.text_input("Write an article about: ")

    def send_to_openai_api(input_data, input_type, prompt, temperature, options):
        # Process input_data and send it to OpenAI API using the prompt
        options_string = ", ".join(options)
        print(options_string)
        custom_instruction = f"] in the style of the provided example, capturing its {options_string}. Example: {input_data}"
        full_prompt = f"{prompt}\n {custom_instruction}\n"

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                    {"role": "user", "content": full_prompt}
                ],
            n=1,
            stop=None,
            temperature=temperature,
        )

        # Return the AI-generated text
        return response.choices[0].message.content

    def main():
        st.title("WriteLikeMe!")
        input_data, input_type, temperature, options = get_input()
        prompt = "Write an article about: " + get_prompt()

        previous_outputs = []
        if "previous_outputs" in st.session_state:
            previous_outputs = st.session_state.previous_outputs

        if st.button("Submit"):
            ai_response = send_to_openai_api(input_data, input_type, prompt, temperature, options)
            st.write(ai_response)
            previous_outputs.append(ai_response)
            st.session_state.previous_outputs = previous_outputs
            with open('previous_outputs.json', 'w') as f:
                json.dump(previous_outputs, f)

        st.sidebar.title("Previous Outputs")
        for i, output in enumerate(previous_outputs):
            with st.sidebar.expander(f"Output {i + 1}"):
                st.write(output)
    def load_previous_outputs():
        if os.path.exists('previous_outputs.json'):
            with open('previous_outputs.json', 'r') as f:
                st.session_state.previous_outputs = json.load(f)
        else:
            st.session_state.previous_outputs = []

    if __name__ == "__main__":
        load_previous_outputs()
        main()