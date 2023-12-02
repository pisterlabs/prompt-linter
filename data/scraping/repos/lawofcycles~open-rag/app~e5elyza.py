import time
import torch
from typing import Optional, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import CTranslate2
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import Generation, LLMResult

# embed model
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

db = FAISS.load_local("faiss_index", embeddings)
question = "カスタマー・デュー・ディリジェンスとはなんですか。"

start = time.time()
# 質問に対して、データベース中の類似度上位3件を抽出。質問の文章はこの関数でベクトル化され利用される
docs = db.similarity_search(question, k=3)
elapsed_time = time.time() - start
print(f"処理時間[s]: {elapsed_time:.2f}")
for i in range(len(docs)):
    print(docs[i])

# setup LLM
MODEL_NAME = "elyza/ELYZA-japanese-Llama-2-7b-fast-instruct"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    do_sample=True,
    top_k=20,
    temperature=0.1,
    # device=device,
)
llm = HuggingFacePipeline(pipeline=pipe)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "参考情報を元に、ユーザーからの質問に簡潔に正確に答えてください。"
text = "{context}\nユーザからの質問は次のとおりです。{question}"
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

# RAG ありの場合
start = time.time()
# ベクトル検索結果の上位3件と質問内容を入力として、elyzaで文章生成
inputs = {"input_documents": docs, "question": question}
output = chain.run(inputs)
elapsed_time = time.time() - start
print("RAGあり")
print(f"処理時間[s]: {elapsed_time:.2f}")
print(f"出力内容：\n{output}")
print(f"トークン数: {llm.get_num_tokens(output)}")

###################################################
# メモリの解放

del model, tokenizer, pipe, llm, chain
torch.cuda.empty_cache()

