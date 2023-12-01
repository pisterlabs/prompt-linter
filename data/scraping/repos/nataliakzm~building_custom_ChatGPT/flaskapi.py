# Import necessary modules
from flask import Flask, request, jsonify

import os
import openai

import tiktoken
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import RetrievalQA   

#Use your own API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

#Initialize the Flask app
app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
        
    #Read more about tokenization in the previous article: "Part 2: An Overview of LLM Development & Training ChatGPT"
    # Specify the model to use and get the appropriate encoding
    tiktoken.encoding_for_model('gpt-3.5-turbo')
    tokenizer = tiktoken.get_encoding('cl100k_base')

    # Load an unstructured PDF file
    loader = UnstructuredPDFLoader('/content/yourdocument.pdf')
    data = loader.load()

    # Define a function to get token length
    def tiktoken_len(text):
            tokens = tokenizer.encode(text, disallowed_special=())
            return len(tokens)

    #Calling chunk splitter from the previous section
    # Split document into chunks using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20,
            length_function=tiktoken_len, separators=["\n\n", "\n", " ", ""])

    # Split the loaded document into chunks
    texts = text_splitter.split_documents(data)

    # Create embeddings 
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai.api_key)

    # Save the embeddings to a FAISS vector store
    vectoredb = FAISS.from_documents(texts, embeddings)

    # Save the vector store locally
    vectoredb.save_local("faiss_index")

    # Load the vector store from the local file
    new_vectoredb = FAISS.load_local("faiss_index", embeddings)

    # Create a retriever from the vector store
    retriever = new_vectoredb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    #Define the instructions for the LLM
    template = """Question: {question}
            Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm_chain = RetrievalQA.from_chain_type(llm=OpenAI(),
                                            chain_type="stuff", 
                                            retriever=retriever,
                                            chain_type_kwargs={"prompt": PROMPT})
              
    
    # Extract question from the incoming JSON request
    data = request.get_json()
    question = data['question'] 

    # Run the question through the model
    answer = llm_chain.run(question)

    # Convert result to JSON and return it
    return jsonify(answer), 200

# Run the app
if __name__ == "__main__":
    app.run(debug=True) 
