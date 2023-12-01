import streamlit as st
import fitz
import tempfile
import openai

# Set your OpenAI API key here
openai.api_key = "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"

# Streamlit app title
st.title("Technical Document Analysis with GPT-3.5 Turbo")

# Upload the technical document (PDF only)
uploaded_file = st.file_uploader("Upload a Technical Document (PDF)", type=["pdf"])

# Check if a file is uploaded
if uploaded_file:
    # Temporary file to store the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Read the uploaded content and extract text from the PDF
    with fitz.open(temp_file_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()

    # Split the document into chunks (you can customize chunk size)
    chunk_size = 2000  # Adjust the chunk size as needed
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Initialize the GPT-3.5 Turbo model from OpenAI
    gpt3_model = "gpt-3.5-turbo"  # Use the GPT-3.5 Turbo model

    # You can now analyze each chunk using GPT-3.5 Turbo and provide recommendations.
    st.subheader("Document Analysis and Recommendations:")
    for i, chunk in enumerate(chunks):
        st.write(f"Chunk {i + 1}:")
        
        # Perform analysis and generate recommendations using GPT-3.5 Turbo
        response = openai.ChatCompletion.create(
            model=gpt3_model,
            messages=[
                {"role": "system", "content": "You are going to identify the inconsistencies, errors, and omissions, and provide effective recommendations to improve this text."},
                {"role": "user", "content": chunk},
            ],
        )

        analysis_result = response.choices[0].message["content"]
        st.write(analysis_result)

    # You can add further post-processing and analysis logic here based on the generated insights.


# Gives Recommendations for improvements, and identifies inconsistencies, errors, and omissions in technical documents.