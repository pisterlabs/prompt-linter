from transformers import AutoTokenizer,AutoModelForCausalLM
import torch

from transformers import pipeline
from langchain.llms import HuggingFacePipeline

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding,download_loader
from typing import Any, List

embed_model_name = "efederici/e5-base-multilingual-4096"
llm_model_name = "lmsys/vicuna-13b-v1.5-16k"

# トークナイザーの初期化
tokenizer = AutoTokenizer.from_pretrained(
    llm_model_name,
    use_fast=True,
)

# LLMの読み込み
model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

# パイプラインの作成
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=4096,
)

# LLMの初期化
llm = HuggingFacePipeline(pipeline=pipe)

# query付きのHuggingFaceEmbeddings
class HuggingFaceQueryEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents(["query: " + text for text in texts])

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query("query: " + text)

# 埋め込みモデルの初期化
embed_model = LangchainEmbedding(
    HuggingFaceQueryEmbeddings(model_name=embed_model_name)
)

from llama_index import SimpleDirectoryReader

# ドキュメントの読み込み
persist_dir = "./resource/211122_amlcft_guidelines.pdf"
CJKPDFReader = download_loader("CJKPDFReader")

loader = CJKPDFReader()
documents = loader.load_data(file=persist_dir)

from llama_index.callbacks import CallbackManager, LlamaDebugHandler
llama_debug_handler = LlamaDebugHandler()
callback_manager = CallbackManager([llama_debug_handler])

from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.prompts.base import ChatPromptTemplate

# QAシステムプロンプト
TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "あなたは世界中で信頼されているQAシステムです。\n"
        "事前知識ではなく、常に提供されたコンテキスト情報を使用してクエリに回答してください。\n"
        "従うべきいくつかのルール:\n"
        "1. 回答内で指定されたコンテキストを直接参照しないでください。\n"
        "2. 「コンテキストに基づいて、...」や「コンテキスト情報は...」、またはそれに類するような記述は避けてください。"
    ),
    role=MessageRole.SYSTEM,
)

# QAプロンプトテンプレートメッセージ
TEXT_QA_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "コンテキスト情報は以下のとおりです。\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "事前知識ではなくコンテキスト情報を考慮して、クエリに答えます。\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]

# チャットQAプロンプト
CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)

# チャットRefineプロンプトテンプレートメッセージ
CHAT_REFINE_PROMPT_TMPL_MSGS = [
    ChatMessage(
        content=(
            "あなたは、既存の回答を改良する際に2つのモードで厳密に動作するQAシステムのエキスパートです。\n"
            "1. 新しいコンテキストを使用して元の回答を**書き直す**。\n"
            "2. 新しいコンテキストが役に立たない場合は、元の回答を**繰り返す**。\n"
            "回答内で元の回答やコンテキストを直接参照しないでください。\n"
            "疑問がある場合は、元の答えを繰り返してください。"
            "New Context: {context_msg}\n"
            "Query: {query_str}\n"
            "Original Answer: {existing_answer}\n"
            "New Answer: "
        ),
        role=MessageRole.USER,
    )
]

# チャットRefineプロンプト
CHAT_REFINE_PROMPT = ChatPromptTemplate(message_templates=CHAT_REFINE_PROMPT_TMPL_MSGS)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index import ServiceContext

text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=4096-3,
    chunk_overlap=20,  # オーバーラップの最大トークン数
    separators=["\n= ", "\n== ", "\n=== ", "\n\n", "\n", "。", "「", "」", "！", "？", "、", "『", "』", "(", ")"," ", ""],
)
node_parser = SimpleNodeParser(text_splitter=text_splitter)

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    node_parser=node_parser,
    callback_manager=callback_manager,
)

from llama_index import VectorStoreIndex

index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context,
)

# クエリエンジンの準備
query_engine = index.as_query_engine(
    similarity_top_k=3,
    text_qa_template=CHAT_TEXT_QA_PROMPT,
    refine_template=CHAT_REFINE_PROMPT,
)

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.WARNING, force=True)
import torch

def query(question):
    print(f"Q: {question}")
    response = query_engine.query(question).response.strip()
    print(f"A: {response}\n")
    torch.cuda.empty_cache()
    
questions = [
    "リスクベースのアプローチとは？",
    "金融機関によるリスク低減措置の具体的内容は？",
    "経営陣はマネロンにどう関与すべき？",
]

for question in questions:
    query(question)

