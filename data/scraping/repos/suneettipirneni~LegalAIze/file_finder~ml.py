
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
import os
from tqdm import tqdm



import os
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import OpenAI
from langchain.chains import AnalyzeDocumentChain, LLMRequestsChain, LLMChain
from langchain.document_loaders import UnstructuredWordDocumentLoader
import os
import redis
import PyPDF2
import pytesseract
from pdf2image import convert_from_path
from langchain.prompts import PromptTemplate
import spacy

# check windows or not
if os.name == 'nt':
    from dotenv import load
    load()
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
else:
    from dotenv import load_dotenv
    load_dotenv()

def get_text_from_pdf(file_path: str):
    text = ""
    pdf_file = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    pdf_file.close()

    # remove all whitespaces and newlines
    text = text.replace('\n', ' ').replace('\r', '').replace('  ', ' ')

    if text == "":
        # extract images from pdf
        images = convert_from_path(file_path)
        for image in images:
            # extract text from image
            text += pytesseract.image_to_string(image)
    return text


def get_text(file_name, file_path):
    text = ""
    if file_name.endswith('.docx') or file_name.endswith('.doc'):
        doc_loader = UnstructuredWordDocumentLoader(file_path)
        document = doc_loader.load()
        text = document[0].page_content
    elif file_name.endswith('.pdf'):
        text = get_text_from_pdf(file_path)
    elif file_name.endswith('.txt'):
        with open(file_path, 'r') as f:
            text = f.read()
    elif file_name.endswith('.jpg') or file_name.endswith('.png'):
        text = pytesseract.image_to_string(file_path)
    else:
        pass
    return text


folder_path = 'E:\\UCF Masters\\Hackathon\\repo\\LegalAIze\\server\\docs'


text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 0,
    length_function = len,
)


embeddings = OpenAIEmbeddings()

def generate_index(folder_path):
    dbs = []
    total_files = len(os.listdir(folder_path))
    count = 0
    for file in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file)
        file_name = file_path.split('/')[-1]
        text = get_text(file_name, file_path)
        if text == "":
            continue
        chunks = text_splitter.create_documents([text], metadatas=[{'file': file_path}])
        db = FAISS.from_documents(chunks, embeddings)
        dbs.append(db)

    # merge all dbs
    merged_db = dbs[0]
    for db in dbs[1:]:
        merged_db.merge_from(db)
    
    merged_db.save_local(os.path.join(folder_path, 'index'))
    return

def query_index(query, db_path):
    db = FAISS.load_local(db_path, embeddings)
    docs_and_scores = db.similarity_search_with_score(query, k=10)
    file_names = [doc[0].metadata['file'] for doc in docs_and_scores]
    return file_names