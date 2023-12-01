import streamlit as st
import openai
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from PyPDF2 import PdfReader
from docx import Document
import os

api_key = "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"
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

# Choose the input source (Text, PDF, or Word)
input_source = st.radio("Choose input source:", ("Text", "PDF", "Word"))

# Prompt user to enter code or select a file
code_input = ""  # Initialize code_input variable

if input_source == "Text":
    code_input = st.text_area("Enter your code:")
elif input_source == "PDF":
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
    if pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            code_input += page.extract_text()
elif input_source == "Word":
    word_file = st.file_uploader("Upload a Word file", type=["docx", "doc"])
    if word_file:
        doc = Document(word_file)
        for paragraph in doc.paragraphs:
            code_input += paragraph.text

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
