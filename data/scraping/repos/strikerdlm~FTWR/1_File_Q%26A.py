import streamlit as st
import openai
from pypdf import PdfReader
from io import BytesIO

st.title("📝 PDF Q> ")

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="file_qa_api_key", type="password")
    model = st.selectbox("Select Model", ["gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-16k", "gpt-4-0613"])
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

uploaded_file = st.file_uploader("Upload an article", type=("pdf"))
question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

if uploaded_file and question and not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")

if uploaded_file and question and openai_api_key:
    # Read PDF file
    pdf_file = PdfReader(BytesIO(uploaded_file.read()))
    article = ' '.join(page.extract_text() for page in pdf_file.pages)
    
    # Set OpenAI API key
    openai.api_key = openai_api_key
    
    # Create prompt for the model
    prompt = f"This is an article:\n\n{article}\n\n{question}"

    # Generate response from the selected model
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        temperature=temperature
    )
    
    st.write("### Answer")
    st.write(response['choices'][0]['message']['content'])
