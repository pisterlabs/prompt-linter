import sentence_transformers
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def init_global_model(model_path):
    print("MODEL_PATH", model_path)
    global_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    global_model = model.eval()
    return global_model, global_tokenizer


def init_embeddings(embedding_model):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model, )
    embeddings.client = sentence_transformers.SentenceTransformer(embeddings.model_name, device=DEVICE)
    return embeddings
#
#
# if USE_LLM_MODEL:
#     global_model, global_tokenizer = init_global_model(MODEL_PATH)
#     # global global_model
#     # global global_model
#
# if USE_EMBEDDING_MODEL:
#     global_embeddings = init_embeddings(EMBEDDING_PATH)
#     # global global_embeddings
