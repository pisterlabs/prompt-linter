
# RUN THIS CELL FIRST!
#/*!pip install -q langchain==0.0.150 pypdf pandas matplotlib tiktoken textract transformers openai faiss-cpu*/
from flask import Flask, request, jsonify, render_template
import os
import textract
import pandas as pd
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from secret import Secret


app = Flask(__name__)

os.environ["OPENAI_API_KEY"] = Secret

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Load and prepare data once, before first request
@app.before_first_request
def prepare_data():
    global chunks, qa, chat_history
    chunks = []  # Initialize the variable before using it
    # Load PDF and split into pages

#     loader = PyPDFLoader("./Urbanite_Deck1.pdf")
#     pages = loader.load_and_split()
#     chunks = pages

#     # Convert PDF to text
#     doc = textract.process("./Urbanite_Deck1.pdf")
    
#     # Save to .txt and reopen
#     with open('Urbanite_Deck1.txt', 'w') as f:
#         f.write(doc.decode('utf-8'))

#     with open('Urbanite_Deck1.txt', 'r') as f:
#         text = f.read()

    
    with open('Urbanite_Deck1.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 256,
        chunk_overlap  = 24,
        length_function = count_tokens,
    )
    chunks += text_splitter.create_documents([text])

    # Embed text and store embeddings
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)

    # Create conversation chain that uses our vectordb as retriever
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.5), db.as_retriever())
    chat_history = []
@app.route('/', methods=['GET', 'POST'])
def ask():
    if request.method == 'POST':
        query = request.form['query']
        result = qa({"question": query, "chat_history": chat_history})
        chat_history.append((query, result['answer']))
        return render_template('ask.html', answer=result['answer'])
    return render_template('ask.html')

@app.route('/welcome')
def home():
    return "Welcome to the Urbanite chatbot!"

@app.route('/api', methods=['POST'])
def chat():
    query = request.json['query']
    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    return jsonify(result['answer'])

if __name__ == '__main__':
    app.run(port=5000, debug=True)
