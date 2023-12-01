import streamlit as st
import openai
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

import os

api_key = os.getenv("OPENAI_API_KEY")
# Set up OpenAI API key
openai.api_key = api_key

# Define function to generate code comments
def generate_comment(query):
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt="/*{}*/".format(query),
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    comment = response.choices[0].text
    return comment

# Create Streamlit UI
st.title("Code Comment Generator")

# Prompt user to enter code
code_input = st.text_area("Enter your code:")

# Determine the programming language of the input code
lexer = get_lexer_by_name("text")
try:
    lexer = get_lexer_by_name(st.session_state["language"])
except:
    for lang in ["python", "java", "javascript", "html", "css"]:
        if lang in code_input.lower():
            lexer = get_lexer_by_name(lang)
            st.session_state["language"] = lang
            break

# Highlight and display the input code
formatter = HtmlFormatter(full=True, style="colorful")
highlighted_code = highlight(code_input, lexer, formatter)
st.write(highlighted_code, unsafe_allow_html=True)

# Generate comments when button is clicked
if st.button("Generate comments"):
    comment = generate_comment(code_input)
    st.code(comment, language=st.session_state["language"])
