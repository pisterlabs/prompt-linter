import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI as OpenAIGPT
from langchain.chains.question_answering import load_qa_chain
import pinecone

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp_uploads'

# Step 0 - Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")
prompt = os.environ.get("OPENAI_PROMPT")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = OpenAIGPT(prompt=prompt,temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm)


@app.route('/')
def index():
  return render_template('index.html')


@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
  uploaded_files = request.files.getlist("pdf_files[]")

  annual_reports = []
  for pdf in uploaded_files:
    filename = secure_filename(pdf.filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf.save(temp_path)
    loader = PyPDFLoader(temp_path)
    document = loader.load()
    annual_reports.append(document)
    os.remove(temp_path)  # Clean up the temporary file

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                 chunk_overlap=0)

  chunked_annual_reports = []
  for annual_report in annual_reports:
    texts = text_splitter.split_documents(annual_report)
    chunked_annual_reports.append(texts)

  index_name = "pdf-chat"

  for chunks in chunked_annual_reports:
    Pinecone.from_texts([chunk.page_content for chunk in chunks],
                        embeddings,
                        index_name=index_name)

  global vectorstore
  vectorstore = Pinecone.from_existing_index(index_name=index_name,
                                             embedding=embeddings)

  return jsonify(success=True)


@app.route('/ask_question', methods=['POST'])
def ask_question():
  query = request.form['question']
  docs = vectorstore.similarity_search(query, include_metadata=True)
  answer = chain.run(input_documents=docs, question=query)

  return jsonify(answer=answer)


if __name__ == '__main__':
  app.run(host="0.0.0.0", port=8080)
