import pytesseract
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.llms import GooglePalm
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json

import google.generativeai as palm
load_dotenv()

llm = GooglePalm()



pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

loader = UnstructuredImageLoader("Sterling Fabory (India) Pvt. Ltd-1.jpg", mode = 'elements')

data = loader.load()

prompt = '''Given is an extracted text from an image.

{text}

Try to find all the entities like Invoice number, invoice date, GST number(CGST, SGST), Vendor name, Vendor address, delivery address, Buyer Name, Buyer address, item details (including prices, quantity, item codes, discount and taxation), total invoice amount, total tax amount, PO number.

* Convert the extracted data into JSON format.

json_output:
'''

prompt_template = PromptTemplate(template=prompt, input_variables = ['text'])

chain = LLMChain(llm=llm, prompt=prompt_template)

json_output = chain.run(data)

print(json_output)



with open('data.json', 'w') as f:
    json.dump(json_output, f)

