# 这里是为了验证模型是否成功加载
# 直接 python valid.py
import sentence_transformers
import torch
# 检查是否能够加载
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from config import *
from chatllm import ChatLLM

EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"
LLM_DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"
#LLM_DEVICE,EMBEDDING_DEVICE are string
num_gpus = torch.cuda.device_count()
print("EMBEDDING_DEVICE: ", EMBEDDING_DEVICE)
print("LLM_DEVICE: ", LLM_DEVICE)
print("num_gpus: ", num_gpus)
print(LLM_DEVICE.lower().startswith("cuda"))



embedding_model_dict = embedding_model_dict
llm_model_dict = llm_model_dict
EMBEDDING_DEVICE = EMBEDDING_DEVICE
LLM_DEVICE = LLM_DEVICE
num_gpus = num_gpus#GPU数量
large_language_model = init_llm
embedding_model=init_embedding_model


try:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model], )
    print('第一步加载成功')
    embeddings.client = sentence_transformers.SentenceTransformer(
                embeddings.model_name,
                device=EMBEDDING_DEVICE,
                cache_folder=os.path.join(MODEL_CACHE_PATH,embeddings.model_name))
    print('embedding模型加载成功')

    llm = ChatLLM()
    if 'chatglm2' in large_language_model.lower():
        llm.model_type = 'chatglm2'
        llm.model_name_or_path = llm_model_dict['chatglm2'][large_language_model]
    llm.load_llm(llm_device=LLM_DEVICE, num_gpus=num_gpus)
    print('LLM加载成功')
except Exception as e:
    print('模型加载失败')

