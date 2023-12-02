import os
import textract

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from pdfminer.high_level import extract_text

import pinecone

textract_config = {
    '.pdf': {
        'pdftotext': None,
        'command': 'pdfminer.six',
    },
}

def convert_to_text_files(data_directory):
    text_files_directory = os.path.join(data_directory, 'text_files')
    if not os.path.exists(text_files_directory):
        os.makedirs(text_files_directory)

    for filename in os.listdir(data_directory):
        input_file_path = os.path.join(data_directory, filename)
        if filename.endswith(".pdf") or filename.endswith(".docx") or filename.endswith(".pptx"):
            try:
                if filename.endswith(".pdf"):
                    text = extract_text(input_file_path)
                elif filename.endswith(".docx") or filename.endswith(".pptx"):
                    text = textract.process(input_file_path, config=textract_config).decode('utf-8')
                else:
                    continue

                output_file_path = os.path.join(text_files_directory, f"{filename}.txt")
                with open(output_file_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(text)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")