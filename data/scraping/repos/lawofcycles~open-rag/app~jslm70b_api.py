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

import logging

# ロガーの設定
logger = logging.getLogger("uvicorn.error")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

app = FastAPI(
    title="Inference API for stabilityai/japanese-stablelm-instruct-beta-70b",
    description="A simple API that use stabilityai/japanese-stablelm-instruct-beta-70b as a chatbot",
    version="1.0",
)

# embed model
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

MODEL_NAME = "stabilityai/japanese-stablelm-instruct-beta-70b"
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,low_cpu_mem_usage=True, trust_remote_code=True)

# Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)
pipe = pipeline(
    "text-generation",
    model=model,
    max_new_tokens=128,
    temperature=0.99,
    top_p=0.95,
    do_sample=True,
    tokenizer=tokenizer,
    repetition_penalty=1.0, 
)
llm = HuggingFacePipeline(pipeline=pipe)

B_INST, E_INST = "<s>[INST] ", " [/INST] "
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """あなたは銀行のQAボットです。対応マニュアルを要約して、質問に答えてください。\n
        以下のルールに従ってください。\n
        - 質問を繰り返さないでください\n
        - 回答に改行を入れてください\n"""
text = "質問:{question}\n対応マニュアル:{context}\n"
template = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
    bos_token=tokenizer.bos_token,
    b_inst=B_INST,
    system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
    prompt=text,
    e_inst=E_INST,
)
rag_prompt_custom = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

# チェーンの準備
chain = load_qa_chain(llm, chain_type="stuff", prompt=rag_prompt_custom)

@app.get('/model')
async def model(question : str):
    logger.info(f"質問：\n{question}")
    start = time.time()
    db = FAISS.load_local("faiss_index/mufgfaq", embeddings)
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