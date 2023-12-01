from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

# Read PDF text
def GetPDFText(pdfDocs):
     pdfText = ""

     for pdfDoc in pdfDocs:
          pdfReader = PdfReader(pdfDoc)
          for pdfPage in pdfReader.pages:
               pdfText += pdfPage.extract_text()

     return pdfText  

# Text separate to chunks
def GetTextChunks(rawText):
     textSplitter = CharacterTextSplitter(
          separator="\n",
          chunk_size=1000,
          chunk_overlap=200,
          length_function=len
     )       
     chunks =textSplitter.split_text(rawText)
     return chunks

