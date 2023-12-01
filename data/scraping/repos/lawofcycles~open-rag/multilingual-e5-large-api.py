import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.WARNING, force=True)
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    download_loader,
    ServiceContext,
    LangchainEmbedding,
    SimpleKeywordTableIndex,
)
from llama_index.vector_stores import FaissVectorStore
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline,
)

# import QueryBundle
from llama_index import QueryBundle

# import NodeWithScore
from llama_index.schema import NodeWithScore

# Retrievers
from llama_index.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)

from typing import List

from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import torch
from llama_index.llms import HuggingFaceLLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain.llms import HuggingFacePipeline
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.prompts.base import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index import ServiceContext
from llama_index.callbacks import CBEventType
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch

from transformers import pipeline
from langchain.llms import HuggingFacePipeline

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding
from typing import Any, List


persist_dir = "./resource/211122_amlcft_guidelines.pdf"

CJKPDFReader = download_loader("CJKPDFReader")

loader = CJKPDFReader()
documents = loader.load_data(file=persist_dir)

embed_model_name = "intfloat/multilingual-e5-large"
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

model_name = "elyza/ELYZA-japanese-Llama-2-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,device_map="auto")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=500,
    # temperature=0.1,
    pad_token_id=tokenizer.eos_token_id,
    # do_sample=True,
    repetition_penalty=1.2,
)

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

CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)

llm = HuggingFacePipeline(pipeline=pipe)

text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=500-3,
    chunk_overlap=20,  # オーバーラップの最大トークン数
    separators=["\n= ", "\n== ", "\n=== ", "\n\n", "\n", "。", "「", "」", "！", "？", "、", "『", "』", "(", ")"," ", ""],
)
node_parser = SimpleNodeParser(text_splitter=text_splitter)

# ServiceContextの準備
service_context = ServiceContext.from_defaults(
    embed_model=embed_model,
    chunk_size=1024,
    node_parser=node_parser,
    llm=llm,
)

from llama_index.callbacks import CallbackManager, LlamaDebugHandler
llama_debug_handler = LlamaDebugHandler()
callback_manager = CallbackManager([llama_debug_handler])

index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context,
)

query_engine = index.as_query_engine(
    similarity_top_k=10,
    text_qa_template=CHAT_TEXT_QA_PROMPT,
)

def query(question):
    print(f"Q: {question}")
    response = query_engine.query(question).response.strip()
    print(f"A: {response}\n")
    torch.cuda.empty_cache()

query("リスクベースのアプローチとは？")

from llama_index.callbacks import CBEventType
print(llama_debug_handler.get_event_pairs(CBEventType.LLM))

# service_context = ServiceContext.from_defaults(
#     llm=llm,
#     embed_model=embed_model,
#     node_parser=node_parser,
#     callback_manager=callback_manager,
# )

# # クエリエンジンの準備
# query_engine = index.as_query_engine(
#     similarity_top_k=3,
#     text_qa_template=CHAT_TEXT_QA_PROMPT,
#     refine_template=CHAT_REFINE_PROMPT,
# )


# def query(question):
#     print(f"Q: {question}")
#     response = query_engine.query(question).response.strip()
#     print(f"A: {response}\n")
#     torch.cuda.empty_cache()




# system_prompt = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"""

# # This will wrap the default prompts that are internal to llama-index
# prompt_string = """\n\n### 指示: \n{query_str}: \n\n\n### 応答"""
# query_wrapper_prompt = PromptTemplate.from_template(prompt_string)

# llm = HuggingFaceLLM(
#     context_window=1024,
#     max_new_tokens=256,
#     generate_kwargs={"temperature": 0.7, "do_sample": False},
#     system_prompt=system_prompt,
#     query_wrapper_prompt=query_wrapper_prompt,
#     tokenizer_name="novelai/nerdstash-tokenizer-v1",
#     model_name="stabilityai/japanese-stablelm-instruct-alpha-7b-v2",
#     device_map="auto",
#     stopping_ids=[50278, 50279, 50277, 1, 0],
#     tokenizer_kwargs={"max_length": 4096},
#     # uncomment this if using CUDA to reduce memory usage
#     # model_kwargs={"torch_dtype": torch.float16}
# )


# # クエリエンジンの作成
# query_engine = index.as_query_engine(
#     similarity_top_k=3  # 取得するチャンク数 (default:2)
# )

# response = query_engine.query("リスクベースのアプローチとは？")
# print(response)

# if not os.path.exists(persist_dir):
#     os.mkdir(persist_dir)
# documents = SimpleDirectoryReader("data").load_data()
# index = GPTVectorStoreIndex.from_documents(documents)
# index.storage_context.persist(persist_dir)



# from langchain.embeddings import HuggingFaceEmbeddings
# from llama_index import GPTVectorStoreIndex, ServiceContext, LangchainEmbedding

# # 埋め込みモデルの準備
# embed_model = LangchainEmbedding(HuggingFaceEmbeddings(
#     model_name="intfloat/multilingual-e5-large"
# ))

# # ServiceContextの準備
# service_context = ServiceContext.from_defaults(
#     embed_model=embed_model
# )

# # インデックスの生成
# index = GPTVectorStoreIndex.from_documents(
#     documents, # ドキュメント
#     service_context=service_context, # ServiceContext
# )




# app = Flask(__name__)
# tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
# model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')

# @app.route("/embeddings", methods=["POST"])
# def get_embeddings():
#     content = request.json
#     input_texts = content["text"]
#     # Tokenize the input texts
#     batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

#     outputs = model(**batch_dict)
#     embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

#     # normalize embeddings
#     embeddings = F.normalize(embeddings, p=2, dim=1)
#     scores = (embeddings[:2] @ embeddings[2:].T) * 100
#     return jsonify({"embeddings": scores.tolist()})

# def average_pool(last_hidden_states: Tensor,
#                  attention_mask: Tensor) -> Tensor:
#     last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
#     return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# if __name__ == "__main__":
#     app.run(debug=True)
