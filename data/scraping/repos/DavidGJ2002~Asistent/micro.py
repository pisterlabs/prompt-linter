from PyPDF2 import PdfReader  # Cambia esto desde PdfFileReader a PdfReader
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain import PromptTemplate
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PyPDF2 import PdfReader  # Asegúrate de importar PdfReader, no PdfFileReader
from langchain.prompts import PromptTemplate
import json  # Importa la biblioteca json


# Configurar clave de API de OpenAI
OPENAI_API_KEY = "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"
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

app = Flask(__name__)
CORS(app)

def create_embeddings(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PdfReader(pdf_file)  # Cambia esto desde PdfFileReader a PdfReader
        text = ""
        for page in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page].extract_text()  # Cambia esto desde extract_text a extractText

        # ... Resto del código ...


        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings()
        embeddings_pdf = FAISS.from_texts(chunks, embeddings)

        return embeddings_pdf

# Cargar el documento
pdf_path = "Usuario.pdf"
embeddings_pdf = create_embeddings(pdf_path)

# Template de la respuesta
prompt_template = """Respona la pregunta con la mayor precision posible utilizando el contexto proporcionado. si la respuesta no esta
                     contenida en el contexto, digamos "La pregunta esta fuera de contexto, 'no me entrenaron para saber eso' " \n\n
                     contexto: \n {context}?\n
                     pregunta: \n {question} \n
                     respuesta: 
                    """
prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

@app.route('/', methods=['GET'])
def home():
    return render_template('SpeechRecognition .html')  # Corregido el nombre del archivo HTML


@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    docs = embeddings_pdf.similarity_search(question)
    llm = OpenAI(model_name="text-davinci-003")
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=question)
        print(f"Response: {response}")
        print(cb)
        print(type(response))  # Imprime el tipo de response
        print(response)  # Imprime el valor de response

    # Usa response directamente como la respuesta
    answer = response

    resp = jsonify(answer=answer)
    print(resp.get_json())  # Imprimir el objeto JSON antes de enviarlo
    return resp



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
