import os
import pinecone
import tempfile
import PyPDF2
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
import time
from langchain.chains import ConversationChain

# Check to see if there is an environment variable with your API keys, if not, use what you put below
OPENAI_API_KEY = st.secrets["openai_api"]
PINECONE_API_KEY = st.secrets["pinecone_api"]
PINECONE_API_ENV = st.secrets["pinecone_env"]

api_key = PINECONE_API_KEY


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

# Initialize Pinecone and set the api_key
pinecone.init(api_key=api_key, environment=PINECONE_API_ENV)
if not pinecone.list_indexes():
    pinecone.create_index(dimension=1536, name='intelpdf', metric='cosine')


index = pinecone.Index("intelpdf", pool_threads=30)
vectorstore = Pinecone(index, embeddings.embed_query, 'text')

chain = load_qa_chain(llm, chain_type="refine")

def query_find(query):
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
    vectorstore.add_texts(page_texts, metadata,batch_size=100)
    time.sleep(2)

def process_pdf(pdf_path):
    print('process pdf')
    # Load the merged PDF file
    loader = PyPDFLoader(pdf_path)
    data = loader.load()

    # Extract the original document name from the pdf_path
    doc_name = os.path.basename(pdf_path)

    # Split the PDF into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=0)
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

def search_openai():
    query = st.session_state.user_query
    if query:
        result = OpenAISearcher(query)
        if result is not None:
            st.subheader("OpenAI Answer:")
            st.write(result)
    else:
        st.warning("Please enter a question.")




# Streamlit app
# ... (Previous code remains the same)

def main():
    st.title("PDF Document Search and Question Answering")
    print('init and checking pinecone')
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
                perform_search(pdf_paths)
                st.session_state.perform_search_done = True  # Set the session state variable to indicate files are uploaded


    # Wait for files to be uploaded and perform_search to be done before showing the query input
    if "perform_search_done" in st.session_state and st.session_state.perform_search_done:
        # Move this line above the "Uploading files..." message
        query = st.text_input("Enter your question:", key="question_input")
        st.session_state.user_query = query  # Save the user's query to session state

        # Create two columns for the buttons and the text
        col1, col2 = st.columns(2)

        # Search OpenAI button in the first column
        if col1.button("Search OpenAI"):
            search_openai()

        # PDF Answer button in the second column
        if col2.button("PDF Answer"):
            con, pgn, doc_name = query_find(query)
            st.subheader("PDF Answer:")
            st.write(con)
            st.subheader("Page Number:")
            st.write(pgn)
            st.subheader('Document Name')
            st.write(doc_name)


if __name__ == "__main__":
    main()
