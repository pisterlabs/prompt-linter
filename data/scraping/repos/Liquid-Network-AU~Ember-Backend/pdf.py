from dotenv import load_dotenv
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from flask_cors import CORS, cross_origin

load_dotenv()

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/answer', methods=['POST'])
@cross_origin()
def answer_question():
    # Load the PDF file
    pdf_path = 'anotate.pdf'
    pdf_reader = PdfReader(pdf_path)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Get user question from the API request
    user_question = request.json['question']

    # Append the additional text to the user question
    appended_question = user_question + ' What file is this part of?'

    # Create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # Perform question answering
    docs = knowledge_base.similarity_search(appended_question)
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=appended_question)

    return jsonify({'answer': response})

if __name__ == '__main__':
    app.run()

# from dotenv import load_dotenv
# from flask import Flask, request, jsonify
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
# from langchain.callbacks import get_openai_callback

# load_dotenv()

# app = Flask(__name__)

# @app.route('/answer', methods=['POST'])
# def answer_question():
#     # Load the PDF files
#     files = ['anotate.pdf', 'next.svg', 'nodeContents.json']
#     chunks = []
#     file_names = []

#     for file_path in files:
#         if file_path.endswith('.pdf'):
#             pdf_reader = PdfReader(file_path)
#             for page in pdf_reader.pages:
#                 chunk_text = page.extract_text()
#                 if chunk_text:
#                     chunks.append(chunk_text)
#                     file_names.append(file_path)
#         else:
#             with open(file_path, 'r') as file:
#                 chunk_text = file.read()
#                 if chunk_text:
#                     chunks.append(chunk_text)
#                     file_names.append(file_path)

#     # Create embeddings
#     embeddings = OpenAIEmbeddings()
#     knowledge_base = FAISS.from_texts(chunks, embeddings)

#     # Get user question from the API request
#     user_question = request.json['question']

#     # Perform question answering
#     docs = knowledge_base.similarity_search(user_question)
#     llm = OpenAI()
#     chain = load_qa_chain(llm, chain_type="stuff")
#     with get_openai_callback() as cb:
#         response = chain.run(input_documents=docs, question=user_question)

#     # Find the file name corresponding to the chunk that gave the answer
#     answer_file_name = None
#     for doc, file_name in zip(docs, file_names):
#         if doc.text in response:
#             answer_file_name = file_name
#             break

#     # Return the answer with the corresponding file name
#     if answer_file_name:
#         response_with_file = f"This answer came from '{answer_file_name}': {response}"
#     else:
#         response_with_file = response

#     return jsonify({'answer': response_with_file})

# if __name__ == '__main__':
#     app.run()