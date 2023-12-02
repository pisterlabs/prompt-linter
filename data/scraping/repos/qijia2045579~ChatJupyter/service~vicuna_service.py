from flask import Flask
from flask import request, send_file
import json
app = Flask(__name__)



def model_load_alpaca_q4():
    from langchain.llms import LlamaCpp
    from langchain import PromptTemplate, LLMChain
    # model_local_location = '/home/jupyter/LLM/models/ggml-alpaca-7b-q4.bin'
    model_local_location = '/home/jupyter/LLM/models/ggml-vic13b-new-q4_0.bin'
    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path=model_local_location, verbose=True,max_tokens = 2048, n_ctx=32768
    )
    template = """Question: {question}

Answer: """

    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    return llm_chain

# llm_chain = model_load_alpaca_q4()

# https://github.com/huggingface/transformers/blob/fe861e578f50dc9c06de33cd361d2f625017e624/src/transformers/pipelines/__init__.py#L506
def model_load_vicuna_7b_hf():
    import os
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    from langchain import HuggingFacePipeline
    from langchain import PromptTemplate,  LLMChain
    llm = HuggingFacePipeline.from_model_id(model_id="TheBloke/Wizard-Vicuna-7B-Uncensored-HF", 
                                            task="text-generation",
                                            device=0, # 0 is for GPU, -1 is for CPU memory
                                            model_kwargs={"temperature":1.3, "max_length":2048}
                                           )
    
    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain

llm_chain = model_load_vicuna_7b_hf()


@app.route("/")
def hello_world():
    answer = llm_chain.run("Hello World")
    return f"<p>{answer}</p>"

# curl -X POST  http://10.128.0.171:9888/post_question -d question='who is Jeo Biden'
@app.route('/post_question', methods=['POST'])
def any_question():
    data = request.form
    question = data['question']
    r = llm_chain.run(question)
    return r


@app.route('/get_app', methods=['GET', 'POST'])
def get_app():
    # uploads = os.path.join(current_app.root_path, app.config['UPLOAD_FOLDER'])
    path = 'app.tar.gz'
    return send_file(path, as_attachment=True)



# app.run(host='0.0.0.0', port=9888,debug=True) # if we set debug to True, it will load the model twice, which will explode the GPU memory
app.run(host='0.0.0.0', port=9888, debug=False)

# Host on ipv6
# app.run(host='::', port=9888,debug=True)