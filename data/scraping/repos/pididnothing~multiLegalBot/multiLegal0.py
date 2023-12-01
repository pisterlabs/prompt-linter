import os
import shutil
from reportlab.pdfgen.canvas import Canvas
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from phospho import Agent, Message
from flask import Flask, request, jsonify
from flask_cors import CORS

os.environ["OPENAI_API_KEY"] = "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"

documents = []
for file in os.listdir('docs'):
    if file.endswith('.pdf'):
        pdf_path = './docs/' + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        doc_path = './docs/' + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        text_path = './docs/' + file
        loader = TextLoader(text_path)
        documents.extend(loader.load())


# we split the data into chunks of 1,000 characters, with an overlap
# of 200 characters between the chunks, which helps to give better results
# and contain the context of the information between chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

# we create our vectorDB, using the OpenAIEmbeddings tranformer to create
# embeddings from our text chunks. We set all the db information to be stored
# inside the ./data directory, so it doesn't clutter up our source files
vectordb = Chroma.from_documents(
  documents,
  embedding=OpenAIEmbeddings(),
  persist_directory='./data'
)
vectordb.persist()

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vectordb.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True
)

# # we can now execute queries against our Q&A chain
# result = qa_chain({'query': 'What are each of the documents about?'})
# print(result['result'])

app = Flask(__name__)
CORS(app)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8095, debug=False)

@app.route('/docqna',methods = ["POST"])
def processclaim():
    try:
        input_json = request.get_json(force=True)
        query = input_json["query"]
        result = qa_chain(query)
        return result['result']
    except:
        return jsonify({"Status":"Failure --- some error occured"})