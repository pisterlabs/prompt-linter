# pythonScript1.py

import sys
import fitz
import PyPDF2
import docx2txt
import textract
import pytesseract
import openai
import os

# Set your OpenAI API key
api_key = "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"

def extract_text_from_pdf(pdf_file):
    text = ''
    pdf_document = fitz.open(pdf_file)
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        text += page.get_text()
    return text

def extract_text_from_docx(docx_file):
    text = docx2txt.process(docx_file)
    return text

def extract_text_from_image(image_file, parser=None):
    text = textract.process(image_file, extension=parser).decode('utf-8')
    return text

def extract_text_from_txt(txt_file):
    with open(txt_file, 'r', encoding='utf-8') as file:
        return file.read()

def generate_summary(text):
    # Initialize the OpenAI API client
    openai.api_key = api_key
    # print("Text before Prompt: ",text)
    # Set the prompt for summarization
    prompt = f"Could you kindly provide an overview of the document's subject matter and its intended purpose along with a log summary?: {text}"
    #prompt = f"What is this document about and what is it leading to? : {text}"
    #prompt = f"How many tokens are present in this text : {text}"

    # Generate the summary using GPT-3.5 Turbo
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        top_p=0.3,
        temperature=0.3,
        max_tokens=400,  # Adjust max_tokens as needed for the desired summary length
        n=1  # Number of completions
    )

    summary = response.choices[0].text.strip()
    print("response.choices[0].text.strip() : ",response.choices[0].text.strip())
    return summary


def reverse_filename(filename):
    # Reverse the filename and find the extension name
    reversed_filename = filename[::-1]
    filename, file_extension = os.path.splitext(reversed_filename)

    # Reverse the extension name back
    file_extension = file_extension[::-1]

    # Reverse the filename again
    filename = filename[::-1]

    return filename, file_extension

def main():
    if len(sys.argv) != 4:
        print("Usage: python pythonScript1.py input_file output_file original_file_name")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    original_file_name = sys.argv[3]

    lol = input_file_path

    # Reverse the filename and find the extension name
    reversed_filename, _ = reverse_filename(original_file_name)
    # print(original_file_name)

    original_file_name = original_file_name[::-1]

    oglen = len(original_file_name)
    lol = lol[:-oglen]
    lol += original_file_name
    # print(lol)
    # Determine the file extension without reversing it
    file_extension = lol.split('.')[-1]

    if file_extension == 'pdf':
        text = extract_text_from_pdf(input_file_path)
    elif file_extension == 'docx':
        text = extract_text_from_docx(input_file_path)
    elif file_extension in ('jpeg', 'png'):
        text = extract_text_from_image(input_file_path,parser=file_extension)
    elif file_extension == 'txt':
        text = extract_text_from_txt(input_file_path)
    else:
        print(f"Unsupported file format for file: {lol}")
        sys.exit(1)

    summary = generate_summary(text)
    print("Summary: ",summary)

    with open(output_file_path, 'w') as output_file:
        output_file.write("Summary: " + summary)



    import shutil

    destination_directory = "final/"  # Replace with the path to your destination directory

    shutil.copy(output_file_path, destination_directory)



    # try:
    #     # Check if the file exists before attempting to delete it
    #     if os.path.exists(input_file_path):
    #         os.remove(input_file_path)
    #     else:
    #         print(f"File '{input_file_path}' does not exist.")
    # except Exception as e:
    #     print(f"Error deleting the file: {e}")

if __name__ == "__main__":
    main()
