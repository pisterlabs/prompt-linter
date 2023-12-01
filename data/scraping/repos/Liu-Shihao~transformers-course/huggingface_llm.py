import os

from flask import Flask, redirect, render_template, request, url_for, jsonify
from langchain import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

app = Flask(__name__)

loader = TextLoader('../example_data/test.txt', encoding='utf8')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()
db = Chroma.from_documents(texts, embeddings)

retriever = db.as_retriever(search_kwargs={"k": 2})

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.15
)

llm = HuggingFacePipeline(pipeline=pipe)
chain = load_qa_chain(llm, chain_type="stuff")


@app.route('/ask', methods=['GET'])
def index():
    # query = request.form["question"]
    query = request.args.get('question')
    print(f"question: {query}")
    result = ask_llm(query)
    print(f"answer: {result}")
    return result


def ask_llm(query):
    docs = retriever.get_relevant_documents(query)
    answer = chain.run(input_documents=docs, question=query)
    return answer


if __name__ == '__main__':
    app.run()
