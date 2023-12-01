# rmm_pdf.py

import os
import re
import openai
from PyPDF2 import PdfReader

# Set your OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

def summarize_text(text, max_tokens=50):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Summarize the following text: {text}",
        max_tokens=max_tokens
    )
    return response.choices[0].text

def clean_line(line):
    str_clean = re.sub('[^a-zA-Z0-9\n\.]', ' ', line)
    return str_clean

def extract_specific_keywords(text):
    keywords = ["machine learning", "model", "supervised"]
    text = text.lower()

    found_keywords = [keyword for keyword in keywords if keyword in text]

    return found_keywords

def extract_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            page_text = clean_line(page_text)
            text += page_text.replace('\n', ' ') + '\n'
            # text += page.extract_text()
    return text

def split_text(text, max_tokens):
    chunks = []
    words = text.split()
    current_chunk = ""
    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_tokens:
            current_chunk += word + " "
        else:
            chunks.append(current_chunk)
            current_chunk = word + " "
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def create_mind_map(pdf_folder, output_folder, max_tokens=4096, subnote_max_tokens=25):
    keyword_nodes = {}

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        pdf_text = extract_text_from_pdf(pdf_path)
        chunks = split_text(pdf_text, max_tokens)
        summarized_text = ""
        # for chunk in chunks:
            # summarized_chunk = summarize_text(chunk, subnote_max_tokens)
            # summarized_text += summarized_chunk
        # print(pdf_file)
        # print(pdf_text)
        # summarized_text = clean_line(summarized_text)  # Clean summarized text
        summarized_text = pdf_text
        keywords = extract_specific_keywords(pdf_text)

        if keywords:
            for keyword in keywords:
                if keyword not in keyword_nodes:
                    keyword_nodes[keyword] = set()
                keyword_nodes[keyword].add(pdf_text)

        output_file = os.path.join(output_folder, f"{pdf_file.replace('.pdf', '.mm')}")

    with open(output_file, "w") as output:
        output.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
        output.write("<map version=\"1.0.1\">\n")

        # Create a main node for the topic
        output.write("  <node TEXT=\"topic\">\n")

        for keyword, subnodes in keyword_nodes.items():
            # Create a subnode for each keyword
            output.write(f"    <node TEXT=\"{keyword}\">\n")
            for subnode in subnodes:
                output.write(f"      <node TEXT=\"{subnode}\" />\n")
            output.write("    </node>\n")

        output.write("  </node>\n")
        output.write("</map>\n")

if __name__ == "__main__":
    pdf_folder = "pdf"  # Folder containing PDF files
    output_folder = "mm"  # Folder to save individual maps
    create_mind_map(pdf_folder, output_folder,  max_tokens=4096, subnote_max_tokens=25)
