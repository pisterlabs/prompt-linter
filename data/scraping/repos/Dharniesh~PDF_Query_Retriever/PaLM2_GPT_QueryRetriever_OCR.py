import os
import pinecone
import tempfile
from PyPDF2 import PdfMerger
import glob
import fitz
from pdf2image import convert_from_path
import streamlit as st
import pytesseract
import shutil
from google.oauth2 import service_account
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI, VertexAI
from langchain.chains import ConversationChain, RetrievalQAWithSourcesChain, LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain import PromptTemplate
import vertexai
from langchain.chains.question_answering import load_qa_chain
import time
import json

# Check to see if there is an environment variable with your API keys, if not, use what you put below
OPENAI_API_KEY = st.secrets["openai_api"]
PINECONE_API_KEY = st.secrets["pinecone_api"]
PINECONE_API_ENV = st.secrets["pinecone_env"]
api_key = PINECONE_API_KEY
#Creating a gcp auth.json file for authenticating with gcp
type = st.secrets['type']
project_id = st.secrets['project_id']
private_key_id = st.secrets['private_key_id']
private_key = st.secrets['private_key']
client_email = st.secrets['client_email']
client_id = st.secrets['client_id']
auth_uri = st.secrets['auth_uri']
token_uri = st.secrets['token_uri']
auth_provider_x509_cert_url = st.secrets['auth_provider_x509_cert_url']
client_x509_cert_url = st.secrets['client_x509_cert_url']
universe_domain = st.secrets['universe_domain']


toml_data = {
    'type': type,
    'project_id': project_id,
    'private_key_id': private_key_id,
    'private_key': private_key,
    'client_email': client_email,
    'client_id': client_id,
    'auth_uri': auth_uri,
    'token_uri': token_uri,
    'auth_provider_x509_cert_url': auth_provider_x509_cert_url,
    'client_x509_cert_url': client_x509_cert_url,
    'universe_domain': universe_domain
}


json_data = json.dumps(toml_data, indent=2)
with open('gcp_auth.json', 'w') as json_file:
    json_file.write(json_data)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "gcp_auth.json"
credentials = service_account.Credentials.from_service_account_file('gcp_auth.json')
PROJECT_ID = "noted-bliss-294509"  
vertexai.init(project=PROJECT_ID, location="us-central1") 
# Initialize Pinecone and set the api_key
pinecone.init(api_key=api_key, environment=PINECONE_API_ENV)
if not pinecone.list_indexes():
    pinecone.create_index(dimension=1536, name='intelpdf', metric='cosine')


index = pinecone.Index("intelpdf", pool_threads=30)
vectorstore = Pinecone(index, embeddings.embed_query, 'text')

# LLM: palm2-bison
llm_palm = VertexAI(
    credentials=credentials,
    model_name="text-bison@001",
    temperature=0.1,
    max_output_tokens=400,    
    verbose=True,
)


def query_find(query,chain):
    print('inside queryfind')
    docsearch = Pinecone.from_existing_index(index_name='intelpdf', embedding=embeddings)
    # Perform the similarity search and get the documents
    docs = docsearch.similarity_search(query=query, k=1)

    # Assuming 'docs' is the list containing your documents or elements
    #print(docs)
    if 'page' in docs[0].metadata:
        pg_ref = [doc.metadata['page'] + 1 for doc in docs]
    else: 
        pg_ref = ['Unavailable']
    if 'source' in docs[0].metadata:  
        doc_name = [doc.metadata['source'] for doc in docs]
    else:
        doc_name= 'Unavailable'

    result = chain.run(input_documents=docs, question=query)

    if pg_ref:
        page_number = pg_ref[0]
        return result, page_number,doc_name
    else:
        print('Nothing to print')


def upload_text_batch(page_texts, metadata):
    print('upload text batch')
    #index.upsert(page_texts)
    vectorstore.add_texts(page_texts, metadata,batch_size=200)
    time.sleep(2)

def imgToPdf(path):
    PDF = pytesseract.image_to_pdf_or_hocr(path, extension='pdf')
    # export to searchable.pdf
    with open("searchable.pdf", "w+b") as f:
        f.write(bytearray(PDF))

def process_pdf(pdf_path):
    print('process pdf')
    # Load the merged PDF file
    loader = PyPDFLoader(pdf_path)
    data = loader.load()

    # Extract the original document name from the pdf_path
    doc_name = os.path.basename(pdf_path)

    # Split the PDF into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

    # Upload the contents and metadata to Pinecone
    metadata = [{"page": t.metadata["page"], "source": doc_name} for t in texts]
    page_texts = [t.page_content for t in texts]

    return page_texts, metadata



# Function to perform the document search and return the results
def perform_search(pdf_paths):
    print('perform search')
    # Check if the file paths exist
    if not all(os.path.exists(path) for path in pdf_paths):
        st.warning("One or more PDF files do not exist.")
        return 'Nothing to Print 1', 'Nothing to Print 2'

    # Initialize the vectorstore once for all batch uploads
    
    #vectorstore = Pinecone(index, embeddings.embed_query, lambda i: metadata[i])

    page_texts_list = []
    metadata_list = []
    for pdf_path in pdf_paths:
        page_texts, metadata = process_pdf(pdf_path)
        page_texts_list.extend(page_texts)
        metadata_list.extend(metadata)

    upload_text_batch(page_texts_list,metadata_list)

def OpenAISearcher(query):
    prompt = '''
    Act as a Question answering agent who assists user's questions :
    '''
    query = query + prompt
    conversation_with_summary = ConversationChain(
    llm=llm, 
    # We set a low k=2, to only keep the last 2 interactions in memory
    memory=ConversationBufferWindowMemory(k=3), 
    verbose=False
)
    return conversation_with_summary.predict(input=query)

def PaLM2Searcher(query):
    prompt = '''
    Act as a Question answering agent who assists user's questions :
    '''
    query = query + prompt
    conversation_with_summary = ConversationChain(
    llm=llm_palm, 
    # We set a low k=2, to only keep the last 2 interactions in memory
    memory=ConversationBufferWindowMemory(k=3), 
    verbose=False
)
    return conversation_with_summary.predict(input=query)

def search_openai():
    query = st.session_state.user_query
    if query:
        result = OpenAISearcher(query)
        if result is not None:
            st.subheader("OpenAI Answer:")
            st.write(result)
    else:
        st.warning("Please enter a question.")

def search_palm2():
    query = st.session_state.user_query
    if query:
        result = PaLM2Searcher(query)
        if result is not None:
            st.subheader("PaLM2 Answer:")
            st.write(result)
    else:
        st.warning("Please enter a question.")

def perform_ocr(image, page_number, output_dir):
    # Perform OCR and extract text from the image
    text = pytesseract.image_to_pdf_or_hocr(image, extension='pdf')
    
    # Save the OCR'd PDF image with a unique filename
    ocr_pdf_filename = os.path.join(output_dir, f'ocr_page_{page_number}.pdf')
    with open(ocr_pdf_filename, 'wb') as f:
        f.write(text)

    return ocr_pdf_filename


def convert_pdf_to_searchable_pdf(input_pdf_file, output_dir, original_filename):
    # Create the "converted" subfolder within the output directory if it does not exist
    converted_output_dir = os.path.join(output_dir, 'converted')
    os.makedirs(converted_output_dir, exist_ok=True)

    # Convert PDF pages to images
    images = convert_from_path(input_pdf_file, output_folder=converted_output_dir)

    # Perform OCR on each image and save as PDF
    ocr_pdf_files = []
    for i, image in enumerate(images):
        ocr_pdf_filename = perform_ocr(image, i, converted_output_dir)
        ocr_pdf_files.append(ocr_pdf_filename)

    # Merge all the OCR'd PDFs into a single searchable PDF
    merger = PdfMerger()
    for ocr_pdf_file in ocr_pdf_files:
        merger.append(ocr_pdf_file)

    # Save the final merged searchable PDF under the "converted" subfolder
    final_output_pdf = os.path.join(converted_output_dir, f'{original_filename}_converted.pdf')
    with open(final_output_pdf, "wb") as f:
        merger.write(f)

    # Remove the temporary extracted images and OCR'd PDFs
    for ocr_pdf_file in ocr_pdf_files:
        os.remove(ocr_pdf_file)
        
def rich_text_identifier(pd_path):
    text = ""
    path = pd_path
    doc = fitz.open(path)
    num = len(doc)//2
    text += doc[num].get_text()
    if text:
        return 1
    else:
        return -1

def doc_validator(file_list):
    output_dir = '.'  # Set the output directory to the current directory
    if not os.path.exists('converted'):
        os.makedirs('converted')

    converted_folder = os.path.join(output_dir, 'converted')
    os.makedirs(converted_folder, exist_ok=True)

    for imgpath in file_list:
        name = imgpath
        val = rich_text_identifier(name)
        if val == -1:
            input_pdf_file = name
            original_filename = os.path.splitext(os.path.basename(name))[0]
            convert_pdf_to_searchable_pdf(input_pdf_file, output_dir, original_filename)
        else:
            # Directly copy the file to the "converted" folder
            shutil.copy(name, os.path.join(converted_folder, os.path.basename(name)))


def main():
    st.title("PDF Document Search and Question Answering")
    print('init and checking pinecone')
    # Dropdown to select between llm and llm_palm
    
    # Initialize Pinecone and set the api_key
    pinecone.init(api_key=api_key, environment=PINECONE_API_ENV)
    if not pinecone.list_indexes():
        pinecone.create_index(dimension=1536, name='intelpdf', metric='cosine')

    # File Upload
    uploaded_files = st.file_uploader("Choose multiple PDF files to merge and search:", type=["pdf"],
                                      accept_multiple_files=True)

    if uploaded_files is not None and len(uploaded_files) >= 1:
        print('inside upload files main')
        # Create a temporary directory to store the uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_paths = []
            for i, file in enumerate(uploaded_files):
                # Get the original filename from the uploaded file
                original_filename = file.name
                # Save the uploaded PDF files to the temporary directory with the original filename
                temp_path = os.path.join(temp_dir, original_filename)
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                pdf_paths.append(temp_path)
    

            if st.button("Upload", key="upload_button"):
                st.text("Uploading files...")
                doc_validator(pdf_paths)
                pdf_paths = glob.glob(os.path.join('converted', '*.pdf'))
                perform_search(pdf_paths)
                st.session_state.perform_search_done = True  # Set the session state variable to indicate files are uploaded
                
    llm_option = st.selectbox("Select LLM for PDF query retrieval:", ["ChatGPT", "PaLM2"])
    if llm_option == "llm":
        selected_llm = llm
    else:
        selected_llm = llm_palm
        
    chain = load_qa_chain(selected_llm, chain_type="refine")


    # Wait for files to be uploaded and perform_search to be done before showing the query input
    #if "perform_search_done" in st.session_state and st.session_state.perform_search_done:
    if os.path.exists('converted'):
        shutil.rmtree("converted")
    # Move this line above the "Uploading files..." message
    query = st.text_input("Enter your question:", key="question_input")
    st.session_state.user_query = query  # Save the user's query to session state
    

    # Create two columns for the buttons and the text
    col1, col2, col3 = st.columns(3)

    # Search OpenAI button in the first column
    if col1.button("Search OpenAI"):
        search_openai()

    # PDF Answer button in the second column
    if col2.button("PDF Answer"):
        con, pgn, doc_name = query_find(query,chain)
        st.subheader("PDF Answer:")
        st.write(con)
        st.subheader("Page Number:")
        st.write(pgn)
        st.subheader('Document Name')
        st.write(doc_name)
    if col3.button("Search PaLM2"):
        search_palm2()


if __name__ == "__main__":
    main()
