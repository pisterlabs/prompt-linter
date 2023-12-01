
import openai
import PyPDF2
import pdfplumber
from langchain.document_loaders import PyPDFLoader

openai.api_key = "API_KEY"

def extract_text_from_pdf(pdf_path):
    pdf_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            
            pdf_text += page.extract_text()
    return pdf_text

def ask_question(question, context):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Read the following text and answer the question:\n{context}\n\nQuestion: {question}\nAnswer:",
        max_tokens=25,
        temperature=0.3,
        stop=None
    )
    answer = response.choices[0].text.strip()
    return answer

loader=PyPDFLoader("Meta-12.31.2022-Exhibit-99.1-FINAL.pdf")
pdf_text = loader.load_and_split()
pdf_text1 = pdf_text[0:5]
pdf_text2 = pdf_text[5:10]
pdf_text3 = pdf_text[10:12]

l=["What are the total assets?","what are total libilities?","What is the net income?","what is the Total stockholders' equity ","what is the total debt?","what is the value of free cash flow at twelve Months Ended December 31,2022","what is the earnings per share at the year end of 2022"]
d={}
p=[pdf_text1,pdf_text2,pdf_text3]
i=0
for pdf_text in p:
    i+=1
    for question in l:
        answer = ask_question(question, pdf_text)
        if 'not' not in answer:
            d[question]=answer
            l.remove(question)
        elif i==len(p):
            d[question]=answer
            l.remove(question)

d["what is the Total stockholders' equity "]="125,713 million"

import re

for key in d.keys():
    input_string = d[key]

    input_string = input_string.replace("$","")
    input_string = input_string.replace(",","")

    numbers = re.findall(r'\d*\.?\d+', input_string)

    if len(numbers) > 0:
        extracted_number = float(numbers[0])
        if "billion" in input_string:
            extracted_number *= 10**9
        elif "million" in input_string:
            extracted_number *= 10**6
        elif "thousand" in input_string:
            extracted_number *= 10**3
        else:
            extracted_number = int(extracted_number)

        d[key]=extracted_number
    else:
        print("No number found in the string.")

ls=[]
for i in d.keys():
    ls.append(d[i])

total_assets = ls[0]
net_income = ls[1]
total_debt = ls[2]
eps = ls[3]
liabilities = ls[4]
free_cash_flow = ls[5]
equity = ls[6]

current_ratio=total_assets/liabilities
return_on_equity=net_income/equity
d_by_e = total_debt/equity
ans={
    "current ratio":current_ratio,
    "return on equity": return_on_equity,
    "Debt by Equity Ratio": d_by_e
}





