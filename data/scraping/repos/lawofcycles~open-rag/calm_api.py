from fastapi import FastAPI, Request
import asyncio
import torch
import time
from transformers import pipeline
from transformers import AutoTokenizer
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate
import copy
from langchain.chains.question_answering import load_qa_chain
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

assert transformers.__version__ >= "4.34.1"
import logging

# ロガーの設定
logger = logging.getLogger("uvicorn.error")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

app = FastAPI(
    title="Inference API for ELYZA",
    description="A simple API that use elyza/ELYZA-japanese-Llama-2-7b-fast-instruct as a chatbot",
    version="1.0",
)

# embed model
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

MODEL_NAME = "cyberagent/calm2-7b-chat"
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto"
)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    do_sample=True,
    temperature=0.1,
    streamer=streamer,
    repetition_penalty=10.0,
)
llm = HuggingFacePipeline(pipeline=pipe)

USER = "USER: "
SYS = "ASSISTANT: "
text = "私の質問に答えるための参考情報として、ユーザの質問に関連するcontextを示します。contextだけを元に質問に答えてください。contextを元に回答できない質問には「わかりません」と答えてください \ncontext:{context}\n質問:{question}\n"
template = "{USER}{text}{SYS}".format(
    USER=USER,
    text=text,
    SYS=SYS,
)

rag_prompt_custom = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

# チェーンの準備
chain = load_qa_chain(llm, chain_type="stuff", prompt=rag_prompt_custom)

@app.get('/model')
async def model(question : str):
    start = time.time()
    db = FAISS.load_local("faiss_index/mufgfaq2", embeddings)
    docs = db.similarity_search(question, k=2)
    elapsed_time = time.time() - start
    logger.info(f"検索処理時間[s]: {elapsed_time:.2f}")
    for i in range(len(docs)):
        logger.info(docs[i])

    start = time.time()
    # ベクトル検索結果の上位3件と質問内容を入力として、elyzaで文章生成
    inputs = {"input_documents": docs, "question": question}
    res = chain.run(inputs)
    result = copy.deepcopy(res)
    elapsed_time = time.time() - start
    logger.info(f"テキスト生成処理時間[s]: {elapsed_time:.2f}")
    logger.info(f"出力内容：\n{result}")
    return result.replace('\n\n', '').replace('\n', '')