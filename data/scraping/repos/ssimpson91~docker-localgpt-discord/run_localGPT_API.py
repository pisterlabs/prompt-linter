import logging
import os
import torch
from flask import Flask, jsonify, request
from run_localGPT import load_model  # Remove load_embeddings from here
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME
from langchain.vectorstores import Chroma

from werkzeug.utils import secure_filename

DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
SHOW_SOURCES = True
logging.info(f"Running on: {DEVICE_TYPE}")
logging.info(f"Display Source Documents set to: {SHOW_SOURCES}")

EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})

# Initialize Flask and Logging
app = Flask(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
)

# Initialize embeddings and database
DB = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=EMBEDDINGS,
    client_settings=CHROMA_SETTINGS,
)

# Initialize LLM
LLM = load_model(DEVICE_TYPE, MODEL_ID, MODEL_BASENAME)

@app.route("/api/receive_user_prompt", methods=["POST"])
def receive_user_prompt():
    user_prompt = request.json.get("user_prompt")
    if user_prompt:
        logging.info(f"Received user prompt: {user_prompt}")

        retriever = DB.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=LLM, chain_type="stuff", retriever=retriever, return_source_documents=SHOW_SOURCES)
        res = qa(user_prompt)
        answer, docs = res["result"], res["source_documents"]

        response = {
            "Prompt": user_prompt,
            "Answer": answer,
            "Sources": [(os.path.basename(str(doc.metadata["source"])), str(doc.page_content)) for doc in docs]
        }

        logging.info(f"Generated response: {response}")
        return jsonify(response), 200

    return jsonify({"error": "No user prompt received"}), 400

if __name__ == "__main__":
    logging.info("Starting Flask server...")
    app.run(debug=False, host='0.0.0.0', port=5110)

