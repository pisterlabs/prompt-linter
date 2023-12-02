# -*- coding: utf-8 -*-
'''
pip install faiss-gpu
pip install faiss-cpu
'''
import os
import torch as th
from transformers import (AutoTokenizer, AutoModel)
from langchain.document_loaders import TextLoader
from langchain.text_splitter import (CharacterTextSplitter, RecursiveCharacterTextSplitter)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import (PromptTemplate, FewShotPromptTemplate)


print(th.cuda.get_device_name())  # NVIDIA GeForce GTX 1080 Ti
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------------------------------------------
# 路径
path_project = "C:/my_project/MyGit/Machine-Learning-Column/LangChain"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# LLM
checkpoint = "chatglm3-6b"  # https://huggingface.co/THUDM/chatglm3-6b

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
    cache_dir=path_model,
    force_download=False,
    local_files_only=True,
    trust_remote_code=True
    )

# GPU
pretrained = AutoModel.from_pretrained(
    pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
    cache_dir=path_model,
    force_download=False,
    local_files_only=True,
    trust_remote_code=True
    ).quantize(4).cuda()
'''
th.cuda.init()
.half.cuda()
'''

# CPU
# pretrained = AutoModel.from_pretrained(
#     pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
#     cache_dir=path_model,
#     force_download=False,
#     local_files_only=True,
#     trust_remote_code=True
#     ).float()

model = pretrained.eval()
# response, history = model.chat(tokenizer, query="你好", history=[])
# response, history = model.chat(tokenizer, query="你能为我做什么？", history=history)

# ----------------------------------------------------------------------------------------------------------------
# doc_loaders
loader = TextLoader(file_path=os.path.join(path_data, "QA.txt"), encoding="utf-8")
docs = loader.load()
type(docs)  # list
type(docs[0])  # langchain.schema.document.Document

# ----------------------------------------------------------------------------------------------------------------
# doc_tranformers
# text_splitter = CharacterTextSplitter(
#     separator="\n",
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len,
#     is_separator_regex=False
# )
text_splitter = RecursiveCharacterTextSplitter(
    separators="\n",
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True
)

documents = text_splitter.split_documents(documents=docs)
# texts = text_splitter.split_text(text=docs[0].page_content)
type(documents)  # list
type(documents[0])  # langchain.schema.document.Document

# ----------------------------------------------------------------------------------------------------------------
# text_embedding_models
# https://huggingface.co/moka-ai/m3e-base
# embedding_model = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base", cache_folder=path_model)

# checkpoint = "text2vec"
checkpoint = "m3e-base"

embedding_model = HuggingFaceEmbeddings(
    model_name=os.path.join(path_model, checkpoint),
    cache_folder=os.path.join(path_model, checkpoint),
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": False}
    )
type(embedding_model)  # langchain.embeddings.huggingface.HuggingFaceEmbeddings

# embeddings = embedding_model.embed_documents(texts=[texts[0].page_content])
# len(embeddings), len(embeddings[0])  # (1, 768)

# embedded_query = embedding_model.embed_query(text="你好！")
# len(embedded_query)  # 768

# ----------------------------------------------------------------------------------------------------------------
# vector_stores
db = FAISS.from_documents(documents=documents, embedding=embedding_model)
# db = FAISS.from_texts(texts=texts, embedding=embedding_model)
type(db)  # langchain.vectorstores.faiss.FAISS

query = "为什么正文大多数用宋体字？"
query = "字体用多少号更合适？"
res_similarity = db.similarity_search(query, k=1)

# res_similarity_score = db.similarity_search_with_score(query, k=1)
# embedded_query = embedding_model.embed_query(query)
# res_similarity = db.similarity_search_by_vector(embedded_query)

# retriever = db.as_retriever()
# type(retriever)  # langchain.schema.vectorstore.VectorStoreRetriever
# res_similarity = retriever.get_relevant_documents(query)

context = "\n".join(res.page_content for res in res_similarity)


# ----------------------------------------------------------------------------------------------------------------
# prompts
# template = "已知信息如下：\n{context}\n根据已知信息回答问题：\n{query}"

template = (
    "你是一个专业的杂志编辑，现在我给你一些已知信息，请你根据这些信息进行回答，答案要标准、语言精炼。\n"
    "已知信息如下：\n{context}\n"
    "根据以上这些信息回答问题：\n{query}"
    )

prompt = PromptTemplate.from_template(template)
prompt_format = prompt.format(context=context, query=query)

type(prompt)  # langchain.prompts.prompt.PromptTemplate
type(prompt_format)  # str

# ----------------------------------------------------------------------------------------------------------------
# chat
response, history = model.chat(tokenizer, query=prompt_format, history=[])






