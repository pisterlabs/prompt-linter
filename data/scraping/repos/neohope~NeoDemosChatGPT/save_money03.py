#!/usr/bin/env python3
# -*- coding utf-8 -*-

import faiss
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTFaissIndex, ServiceContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from llama_index.node_parser import SimpleNodeParser
from langchain.llms.base import LLM
from llama_index import LLMPredictor
from typing import Optional, List, Mapping, Any
from llama_index import QuestionAnswerPrompt
from llama_index import QueryMode
from transformers import AutoTokenizer, AutoModel

"""
通过HuggingFacebEmbedding和sentence-transformers计算向量，并保持到faiss中
通过ChatGLM模型，实现问答
"""

# 封装LLM
class CustomLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response, history = model.chat(tokenizer, prompt, history=[])
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": "chatglm-6b-int4"}

    @property
    def _llm_type(self) -> str:
        return "custom"


if __name__ == '__main__':
    # 加载文件
    documents = SimpleDirectoryReader('./data/faq/').load_data()

    # 句子拆分
    text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=100, chunk_overlap=20)
    parser = SimpleNodeParser(text_splitter=text_splitter)
    nodes = parser.get_nodes_from_documents(documents)

    # 计算向量
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ))

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True)
    # 加载GPU模型
    model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).half().cuda()
    # 引入自定义llm
    llm_predictor = LLMPredictor(llm=CustomLLM(tokenizer, model))
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm_predictor=llm_predictor)

    # 向量放到Faiss中
    dimension = 768
    faiss_index = faiss.IndexFlatIP(dimension)
    index = GPTFaissIndex(nodes=nodes, faiss_index=faiss_index, service_context=service_context)

    # 进行提问
    QA_PROMPT_TMPL = (
        "{context_str}"
        "\n\n"
        "根据以上信息，请回答下面的问题：\n"
        "Q: {query_str}\n"
        )
    QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

    response = index.query(
        "请问你们海南能发货吗？", 
        mode=QueryMode.EMBEDDING,
        text_qa_template=QA_PROMPT,
        verbose=True, 
    )
    print(response)
