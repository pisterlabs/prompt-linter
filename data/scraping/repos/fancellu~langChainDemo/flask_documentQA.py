from flask import Flask, render_template, request, jsonify
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import OpenAI

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()  # Get the JSON data from the request body
    user_input = data.get('input')  # Extract the 'input' field from the JSON data

    # Process the user input and generate a bot response using your existing code
    bot_response = chain.run(user_input)

    # Return the bot response as a JSON response
    response_data = {'reply': bot_response}
    return jsonify(response_data)


if __name__ == "__main__":
    loader = TextLoader('./2020_state_of_the_union.txt', encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)

    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    loader = PyPDFLoader("cats.pdf")
    pages = loader.load_and_split()

    store = Chroma.from_documents(texts + pages, embeddings, collection_name="2020_state_of_the_union_cats")

    llm = OpenAI(temperature=0)
    chain = RetrievalQA.from_chain_type(llm, retriever=store.as_retriever())

    app.run()
