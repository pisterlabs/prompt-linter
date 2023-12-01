import streamlit as st
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import cohere as ch
import base64
@st.cache_data
def convert_pdf_to_txt_pages(path):
    texts = []
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    # fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    size = 0
    c = 0
    file_pages = PDFPage.get_pages(path)
    nbPages = len(list(file_pages))
    for page in PDFPage.get_pages(path):
      interpreter.process_page(page)
      t = retstr.getvalue()
      if c == 0:
        texts.append(t)
      else:
        texts.append(t[size:])
      c = c+1
      size = len(t)
    device.close()
    retstr.close()
    
    return texts, nbPages

def generate_script(user_input):
    co = ch.Client('COHERE_APIKEY') # This is your trial API key (Paste the cohere API KEY)
    response = co.generate(
        model='command',
        prompt=user_input,
        max_tokens=300,
        temperature=0.9,
        k=0,
        stop_sequences=[],
        return_likelihoods='NONE')
    return response.generations[0].text
@st.cache_data
def convert_pdf_to_txt_file(textarea,path):
    texts = []
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    # fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    
    file_pages = PDFPage.get_pages(path)
    nbPages = len(list(file_pages))
    for page in PDFPage.get_pages(path):
      interpreter.process_page(page)
      t = retstr.getvalue()
    # text = retstr.getvalue()

    # fp.close()
    device.close()
    retstr.close()
    t=summarize_script(t)
    
    t=generate_script(textarea+t)
    return t, nbPages
def summarize_script(output1):
    co = ch.Client('COHERE_APIKEY')# This is your trial API key (Paste the cohere API KEY)
    response = co.summarize( 
        text=output1,
        length='auto',
        format='auto',
        model='summarize-xlarge',
        additional_command='',
        temperature=0.3,
    ) 
    return response.summary
@st.cache_data
def save_pages(pages):
  
  files = []
  for page in range(len(pages)):
    filename = "page_"+str(page)+".txt"
    with open("./file_pages/"+filename, 'w', encoding="utf-8") as file:
      file.write(pages[page])
      files.append(file.name)

def displayPDF(file):
  # Opening file from file path
  # with open(file, "rb") as f:
  base64_pdf = base64.b64encode(file).decode('utf-8')

  # Embedding PDF in HTML
  pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
  # Displaying File
  st.markdown(pdf_display, unsafe_allow_html=True)
