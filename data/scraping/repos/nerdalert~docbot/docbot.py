import os
from flask import Flask, render_template, request, redirect
from llama_index import download_loader
from llama_index import LLMPredictor, GPTSimpleVectorIndex, PromptHelper, ServiceContext
from langchain import OpenAI

os.environ["OPENAI_API_KEY"] = "insert_your_key_here"

current_script_path = os.path.dirname(os.path.abspath(__file__))
doc_path = os.path.join(current_script_path, 'data') + '/'

index_file = 'index.json'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = doc_path
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB max file size


def send_click(prompt):
    if index is None:
        return "Index not loaded. Please upload a file first."
    response = index.query(prompt)
    return response


index = None


@app.route('/', methods=['GET', 'POST'])
def index_page():
    global index
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

        SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
        loader = SimpleDirectoryReader(doc_path, recursive=True, exclude_hidden=True)
        documents = loader.load_data()

        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo"))
        max_input_size = 4096
        num_output = 256
        max_chunk_overlap = 20
        prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

        index = GPTSimpleVectorIndex.from_documents(
            documents, service_context=service_context
        )

        index.save_to_disk(index_file)

    elif os.path.exists(index_file):
        index = GPTSimpleVectorIndex.load_from_disk(index_file)

    return render_template('index.html')


@app.route('/query', methods=['POST'])
def query():
    if index is None:
        return {'response': 'Index not loaded. Please upload a file first.'}
    prompt = request.form['prompt']
    response = send_click(prompt)
    return {'response': str(response)}


if __name__ == '__main__':
    # Run flask with the following defaults
    app.run(debug=True, port=5000, host='0.0.0.0', )
