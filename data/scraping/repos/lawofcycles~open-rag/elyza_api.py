import time
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from fastapi import FastAPI
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

import logging
logger = logging.getLogger("uvicorn.error")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

app = FastAPI(
    title="Inference API for ELYZA",
    description="A simple API that with intfloat/multilingual-e5-large and elyza/ELYZA-japanese-Llama-2-7b-fast-instruct",
    version="1.0",
)
# embed model
embed_model_id = "intfloat/multilingual-e5-large"
embeddings = HuggingFaceEmbeddings(model_name=embed_model_id)

# text generation model
model_id = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        use_cache=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=2048,
    do_sample=False,
    top_k=500,
    top_p=0.95,
    temperature=1,
    repetition_penalty=1.05,
)
llm = HuggingFacePipeline(pipeline=pipe)

# Prompt
# For prompt format, see https://huggingface.co/elyza/ELYZA-japanese-Llama-2-7b-fast
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """
You are an AI assistant, helping bank customer by providing answers and advice. \n
Use only information provided in the following context to answer the question at the end.\n
Explain your answer with reference to context as detail as possible.\n
If you cannot answer a user's question based on context, just say "I don't know".\n
Do not preface your answer with a response.\n"""

QUERY = """質問: {question}\n
        context: {context}\n"""
template = f"{tokenizer.bos_token}{B_INST} {B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}{QUERY} {E_INST} "

rag_prompt_custom = PromptTemplate(
    template=template, input_variables=["question","context"]
)

chain = load_qa_chain(llm, chain_type="stuff", prompt=rag_prompt_custom)

@app.get('/query')
async def query(question : str):
    logger.info("質問：\n%s", question)
    # ベクトル検索
    start = time.time()
    db = FAISS.load_local("faiss_index/mufgfaq3", embeddings)
    docs = db.similarity_search(question, k=1)
    search_time = time.time() - start
    logger.info("テキスト生成処理時間[s]: %.2f", search_time)
    logger.info("検索結果:")
    for _, doc in enumerate(docs):
        logger.info(doc)
    # テキスト生成
    start = time.time()
    inputs = {"input_documents": docs, "question": question}
    res = chain.run(inputs)
    result = copy.deepcopy(res)
    generation_time = time.time() - start
    logger.info("テキスト生成処理時間[s]: %.2f", generation_time)
    logger.info("テキスト生成結果:\n%s", result)
    return {"message": result,
            "vector_search_result": docs,
            "search_time": search_time,
            "generation_time": generation_time}