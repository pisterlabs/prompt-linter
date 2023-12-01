import openai
import fitz  # PyMuPDF
import os

# Path to file containing API key
api_key_file_path = "openai_key"

with open(api_key_file_path, 'r') as f:
    openai.api_key = f.read().strip()

def read_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def summarize_text_with_openai(text, chunk_len=2000, temperature=0.2):
    prompt_chunks = [text[i:i + chunk_len] for i in range(0, len(text), chunk_len)]
    summarized_text = ""
    for chunk in prompt_chunks:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a summarization assistant."},
                    {"role": "user", "content": f"Summarize the following text:\n{chunk}"}
                ],
                max_tokens=min(4096, chunk_len),
                temperature=temperature
            )
            summarized_text += response['choices'][0]['message']['content'].strip() + "\n"
        except openai.error.OpenAIError as e:
            print(f"An error occurred: {str(e)}")
    return summarized_text

pdf_path = "your_file.pdf"
text = read_pdf(pdf_path)
if text:
    summarized_text = summarize_text_with_openai(text)
    print(summarized_text)
else:
    print("Failed to extract text from the PDF.")
