import torch
from datasets import Dataset
from flask import Flask, request, jsonify
from flask_cors import CORS
from tqdm import tqdm
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import os

app = Flask(__name__)
CORS(app)

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

template = """Context: {context}

Answer: {answer}"""

prompt = PromptTemplate(template=template, input_variables=["context", "answer"])

model_id = 'shnl/ViT5-vinewqg'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=256
)

local_llm = HuggingFacePipeline(pipeline=pipe)


llm_chain = LLMChain(prompt=prompt,
                     llm=local_llm
                     )

@app.route('/gen', methods=['POST'])
def generate_question():

    data = request.get_json()

    context = data.get('context', '')
    answer = data.get('answer', '')

    return jsonify({'prediction': llm_chain.run(context=context, answer=answer)})

if __name__ == '__main__':
    app.run(port=7777)
