import torch.cuda
import torch.backends
import os

embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec-base": "shibing624/text2vec-base-chinese",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
}

# Embedding model name
EMBEDDING_MODEL = "text2vec"

# Embedding running device
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# supported LLM models
llm_model_dict = {
    "chatyuan": "ClueAI/ChatYuan-large-v2",
    "chatglm-6b-int4-qe": "THUDM/chatglm-6b-int4-qe",
    "chatglm-6b-int4": "THUDM/chatglm-6b-int4",
    "chatglm-6b-int8": "THUDM/chatglm-6b-int8",
    "chatglm-6b": "/home/zhurunqiu/chatgpt/chatglm-6b",
    "chatgpt-3.5" : "chatgpt-3.5"
}

# LLM model name
LLM_MODEL = "chatglm-6b"
LLM_MODEL = "chatgpt-3.5"

# Use p-tuning-v2 PrefixEncoder
USE_PTUNING_V2 = False

OPENAI_API_KEY_IRENSHI = "cuda_asidhoewbhfojr"

# LLM running device
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

VS_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store", "")

# vs_path = None
vs_path = "/home/zhurunqiu/chatgpt/chatglm-demo/langchain-ChatGLM/vector_store/7moor_FAISS_20230503_103526"

UPLOAD_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "content", "")

# 基于上下文的prompt模版，请务必保留"{question}"和"{context}"
PROMPT_TEMPLATE = """基于以下已知信息，简洁和专业的来回答用户的问题，问题是"{question}"。如果无法从中得到答案，请说 "已知信息无法回答该问题"，不允许在答案中添加编造成分，答案请使用中文。已知内容如下: 
{context} """

# 匹配后单段上下文长度
CHUNK_SIZE = 1000

# 进行基本的句相关度检查
min_score = 5000