import os
import textract
import openai
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from pdfminer.high_level import extract_text
from src.constants import OPENAPI_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME

VERBOSE = True
NEW_DATA = False
textract_config = {
    '.pdf': {
        'pdftotext': None,
        'command': 'pdfminer.six',
    },
}

openai.api_key = OPENAPI_KEY
def convert_to_text_files(data_directory):
    text_files_directory = os.path.join(data_directory, 'text_files')
    if not os.path.exists(text_files_directory):
        os.makedirs(text_files_directory)

    for filename in os.listdir(data_directory):
        input_file_path = os.path.join(data_directory, filename)
        if filename.endswith(".pdf") or filename.endswith(".docx") or filename.endswith(".pptx"):
            try:
                if filename.endswith(".pdf"):
                    text = extract_text(input_file_path)
                elif filename.endswith(".docx") or filename.endswith(".pptx"):
                    text = textract.process(input_file_path, config=textract_config).decode('utf-8')
                else:
                    continue

                output_file_path = os.path.join(text_files_directory, f"{filename}.txt")
                with open(output_file_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(text)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    return text_files_directory

def ingestion_engine():
    data_directory = os.path.join(os.getcwd(), 'data')
    text_files_directory = convert_to_text_files(data_directory)

    loader = DirectoryLoader(text_files_directory)
    data = loader.load()
    
    print(f'You have {len(data)} document(s) in your data')
    print(f'There are {len(data[0].page_content)} characters in your document')
    if VERBOSE:
        print(data)
    embeddings  = OpenAIEmbeddings(openai_api_key=OPENAPI_KEY)
    pinecone.init(      
        api_key=PINECONE_API_KEY,      
        environment=PINECONE_ENVIRONMENT,      
    )      
    index_name = PINECONE_INDEX_NAME

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=20)
    texts = text_splitter.split_documents(data)

    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

def query_pinecone(question):
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    index = pinecone.Index(PINECONE_INDEX_NAME)

    prompt = f"Translate the following question into a query: {question}"
    response = openai.Completion.create(prompt=prompt, engine="text-davinci-003", max_tokens=50)
    query = response.choices[0].text
    if VERBOSE:
        print(f'Query: {query}')

    result = index.query(namespace=PINECONE_INDEX_NAME,vector=response, top_k=10)

    return result

def ask_openai_with_pinecone(question):
    result = query_pinecone(question)
    context = result[0].text
    answer = ask_openai(question, context)
    return answer

def ask_openai(question, context):
    openai = OpenAI(openai_api_key=OPENAPI_KEY)
    answer = openai.answer(question, context)
    return answer

def main():
    question = "What is java?"
    if NEW_DATA:
        ingestion_engine()
    ask_openai_with_pinecone(question)
    
if __name__ == "__main__":
    main()
