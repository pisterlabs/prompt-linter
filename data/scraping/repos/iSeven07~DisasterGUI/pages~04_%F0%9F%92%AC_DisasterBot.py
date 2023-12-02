import streamlit as st
import openai
from streamlit_chat import message
from streamlit_extras.app_logo import add_logo
import os

st.set_page_config(page_title="NDD - DisasterBot",
                   page_icon="💬", layout="wide")

add_logo("images/lrw-color.png")

st.title("💬 Chat with DisasterBot")
st.markdown("Meet DisasterBot; your very own AI bot that will answer any question about U.S. disasters that you can ask in plain text.")
st.markdown("#")

try:
  openai.api_key = os.environ["OPENAI_API_KEY"]
except KeyError:
  st.write("API KEY was not found!")

def generate_response(prompt):
  completions = openai.Completion.create(
    engine = "text-davinci-003",
    prompt = prompt,
    max_tokens = 100, #Limited for testing
    n = 1,
    stop = None,
    temperature=0.5,
  )
  message = (completions.choices[0].text).lstrip()

  return message

# Storing Chat Information
if 'generated' not in st.session_state:
  st.session_state['generated'] = []

if 'past' not in st.session_state:
  st.session_state['past'] = []

# Input Area
def get_text():
  input_text = st.text_input("Prompt: ", key="input")
  return input_text
user_input = get_text()

# Get Response from Prompt
if user_input:
  output = generate_response(user_input)
  # Store the output
  st.session_state.past.append(user_input)
  st.session_state.generated.append(output)


# Display Chat History

if st.session_state['generated']:

  for i in range(len(st.session_state['generated']) -1, -1, -1):
    message(st.session_state['generated'][i], key=str(i))
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

else:
  message("Ask me a question about disasters. I'm happy to help.")

# ---- STREAMLIT STYLE ----
st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(st_style, unsafe_allow_html=True)