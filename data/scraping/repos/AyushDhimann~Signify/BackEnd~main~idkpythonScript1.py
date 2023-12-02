# pythonScript1.py

import sys

new_content = ""

def main():
    if len(sys.argv) != 4:
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    original_file_name = sys.argv[3]

    # # Read the original content from the input file
    # with open(input_file_path, 'r') as input_file:
    #     original_content = input_file.read()

    # # Add "line 2" to the original content
    # new_content = original_content + "\nTHIS LINE WAS ADDED INTO THIS FILE FIRST"



    # START

    import fitz
    import os
    import PyPDF2
    import docx2txt
    import textract
    import pytesseract
    from PIL import Image
    import openai

    # Set your OpenAI API key
    api_key = "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"

    # Function to extract text from a PDF file
    def extract_text_from_pdf(pdf_file):
        text = ''
        pdf_document = fitz.open(pdf_file)
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text += page.get_text()
        return text

    # Function to extract text from a Word document
    def extract_text_from_docx(docx_file):
        text = docx2txt.process(docx_file)
        return text

    # Function to extract text from an image (JPEG)
    def extract_text_from_image(image_file):
        text = textract.process(image_file).decode('utf-8')
        return text

    # Function to generate a document summary using GPT-3.5 Turbo
    def generate_summary(text):
        # Initialize the OpenAI API client
        openai.api_key = api_key

        # Set the prompt for summarization
        prompt = f"Please provide a detailed  summary of this document along with name and other important stuff.: {text}"# Also provide 3 questions and it's respective answer related to the document.: {text}"

        # Generate the summary using GPT-3.5 Turbo
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            top_p=0.3,
            max_tokens=150,  # Adjust max_tokens as needed for the desired summary length
            n = 1 # Number of completions
        )

        summary = response.choices[0].text.strip()
        return summary

    # Main function
    def main():
        input_file = input_file_path #"/content/sodapdf-converted.pdf"  # Replace with the path to your input file

        if input_file.endswith('.pdf'):
            text = extract_text_from_pdf(input_file)
        elif input_file.endswith('.docx'):
            text = extract_text_from_docx(input_file)
        elif input_file.endswith('.jpeg'):
            text = extract_text_from_image(input_file)
        elif input_file.endswith('.png'):
            text = extract_text_from_image(input_file)
        else:
            print("Unsupported file format")
            return

        summary = generate_summary(text)
        print(summary)
        new_content = summary

    # Write the updated content to the output file
    with open(output_file_path, 'a+') as output_file:
        output_file.write(new_content)

if __name__ == "__main__":
    main()
