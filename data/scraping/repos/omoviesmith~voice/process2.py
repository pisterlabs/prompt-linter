import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
import sys

# Set the console output encoding to utf-8
sys.stdout.reconfigure(encoding="utf-8")


#Step 1. Load
def load_document(file):
    from langchain.document_loaders import PyPDFLoader
    print("Loading {file} document...")
    loader = PyPDFLoader(file)
    data = loader.load()
    print("Document loaded.")
    return data

#Hybrid loader
def load_documents():
    from langchain.document_loaders import PyPDFDirectoryLoader, WebBaseLoader

    # Load PDF documents from the "pdf" folder
    pdf_folder_path = "pdf"  # Update this to the correct folder path
    pdf_loader = PyPDFDirectoryLoader(pdf_folder_path)
    print("Loading PDF files as documents...")
    pdf_documents = pdf_loader.load()

    if not pdf_documents:
        print("No PDF files found in the 'pdf' folder.")
    else:
        print("PDF Documents loaded.")

    # Load webpages as documents from the "links/url.txt" file
    links_folder = "links"
    url_file = "url.txt"
    webpage_documents = []
    try:
        url_path = os.path.join(links_folder, url_file)
        with open(url_path, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
        if not urls:
            print("No URLs found in the 'links/url.txt' file.")
        else:
            print("Loading webpages as documents...")
            web_loader = WebBaseLoader(urls)
            webpage_documents = web_loader.load()
            print("Webpage Documents loaded.")
    except FileNotFoundError:
        print("The 'links/url.txt' file was not found.")

    # Merge and return the loaded documents
    all_documents = pdf_documents + webpage_documents
    return all_documents




#Step 2. Split
def chunk_data(data, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    print("Chunking data...")
    chunks = text_splitter.split_documents(data)
    return chunks


def print_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'You have {total_tokens} tokens in your document.')
    print(f'Embedding cost: {total_tokens / 1000 * 0.0004:.6f} ADA')

#Step 3. Store
def insert_or_fetch_embeddings(index_name):
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()

    pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment=os.environ.get("PINECONE_ENV"))

    #index_name = 'askadocument'

    if index_name in pinecone.list_indexes():
        print(f'Index {index_name} already exists. Loading embeddings ...', end=' ')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Done.')
    else:
        print(f'Index {index_name} does not exist. Creating index ...', end=' ')
        pinecone.create_index(index_name, dimension=1536, metric='cosine')
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('Done.')
    return vector_store

def ask_and_get_answer(vector_store, q):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI 

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_params={'k': 3})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.run(q)
    return answer

def ask_with_memory2(vector_store, question, chat_history=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})

    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({'question': question, 'chat_history': chat_history})
    chat_history.append((question, result['answer']))

    return result, chat_history

def ask_with_memory(vector_store, question, chat_history=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts.prompt import PromptTemplate
    
    custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. If you do not know the answer reply with 'I am sorry, I dont have this answer'.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
    custom_prompt = PromptTemplate.from_template(custom_template)

    llm = ChatOpenAI(temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':5})

    crc = ConversationalRetrievalChain.from_llm(llm, retriever, condense_question_prompt=custom_prompt)
    result = crc({'question': question, 'chat_history': chat_history})
    chat_history.append((question, result['answer']))

    return result, chat_history



def delete_pinecone_index(index_name):
    import pinecone
    pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment=os.environ.get("PINECONE_ENV"))

    if index_name == 'all':
        indexes = pinecone.list_indexes()
        print(f'Deleting all indexes')
        for index in indexes:
            pinecone.delete_index(index)
        print('Done.')
    else:
        print(f'Deleting index {index_name} ...', end=' ')
        pinecone.delete_index(index_name)
        print('Done.')
    


if __name__ == "__main__":
    # Get the secret key from the environment variables
    # secret_key = os.environ.get("PINECONE_API_KEY")

    # Check if the secret key exists and print it
    # if secret_key:
    #     print(f"Connected! Your secret key is: {secret_key}")
    # else:
    #     print("Secret key not found in the .env file.")
    
    # Load the document
    # # data = load_document('./page_10.pdf')
    # all_documents = load_documents()
    # print(all_documents[0].page_content)

    # # Print the number of pages in the document
    # print(f'You have {len(all_documents)} pages in your document.')

    # # Chunk the data
    # chunks = chunk_data(all_documents)

    # #Delete all indexes
    # delete_pinecone_index('all')
    # # Print the number of chunks in the document
    # print(f'You have {len(chunks)} chunks in your document.')

    # print_embedding_cost(chunks)

    # Insert or fetch embeddings
    index_name = 'thevoice'
    vector_store = insert_or_fetch_embeddings(index_name)

    # Asking with memory
    chat_history = []
    question = 'What is the Uluru Statement From the Heart ?'
    result, chat_history = ask_with_memory(vector_store, question, chat_history)
    print(result['answer'])
    print(chat_history)

    question1 = "how is it linked to the Voice??"
    result, chat_history = ask_with_memory(vector_store, question1, chat_history)
    print(result['answer'])
    print(chat_history)






