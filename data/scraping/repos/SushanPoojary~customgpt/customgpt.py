from flask import Flask, jsonify, request
from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
from flask_cors import CORS
import os

os.environ["OPENAI_API_KEY"] = '' #add your OPENAI API key here
docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

app = Flask(__name__)
CORS(app)

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
    documents = SimpleDirectoryReader(directory_path).load_data()
    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index.save_to_disk('index.json')
    return index


@app.route('/chat', methods=['POST'])
def chatbot():
    input_text = request.json['message']
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return jsonify({'response': response.response})

if __name__ == '__main__':
    index = construct_index(docs_dir)
    app.run(host="0.0.0.0", port=5001, debug=True)