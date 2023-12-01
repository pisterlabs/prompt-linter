
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

ALLOWED_EXTENSIONS = {'txt', 'pdf'}


class Processor:
    def allowed_file(self, filename):
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def get_pdf_text(self, pdf):
        text = ''
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        return text

    def get_text_chunks(self, txt):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n'],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        return text_splitter.split_text(txt)


processor = Processor()
