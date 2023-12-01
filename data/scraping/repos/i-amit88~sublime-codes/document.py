import openai
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

openai.api_key = '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'




def perform_ocr(document_path):
    document_image = Image.open(document_path)

    extracted_text = pytesseract.image_to_string(document_image)

    return extracted_text



def perform_ocr_on_pdf(document_path):
    images = convert_from_path(document_path)

    extracted_text = ''
    for image in images:
        text = pytesseract.image_to_string(image)
        extracted_text += text

    return extracted_text




def scan_document(document):

    extracted_text = perform_ocr(document)
    extracted_text = perform_ocr_on_pdf(document)

    return extracted_text

def answer_question(question, context):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=f"Question: {question}\nContext: {context}\nAnswer:",
        max_tokens=100
    )

    answer = response.choices[0].text.strip()

    return answer

document = 'sample.pdf'  
question = "What is the main topic discussed in the document?"

extracted_text = scan_document(document)

answer = answer_question(question, extracted_text)

print("Answer:", answer)








