import time
import tqdm
import torch
from openai.embeddings_utils import get_embedding
from src import datasets_loader
from src.utils import retrieval_eval

CSN_GFG_DATA_PATH = "/mnt/colab_public/datasets/joao/CodeSearchNet"

EMBEDDING_MODEL_ID = "text-embedding-ada-002"
MAX_RAW_LEN = 15000
SLEEP_SECONDS_BETWEEN_QUERIES = 1.0

test_data = datasets_loader.get_dataset(
    dataset_name="code_search_net",
    path_to_cache=CSN_GFG_DATA_PATH,
    split="validation",
    maximum_raw_length=MAX_RAW_LEN,
)

source_embeddings_list = []
target_embeddings_list = []
total_embeddings = 0

for (source, target) in tqdm.tqdm(test_data, total=len(test_data), desc="embedding"):

    source_embedding = torch.Tensor(get_embedding(source, engine=EMBEDDING_MODEL_ID))[
        None, :
    ]
    target_embedding = torch.Tensor(get_embedding(target, engine=EMBEDDING_MODEL_ID))[
        None, :
    ]

    source_embeddings_list.append(source_embedding)
    target_embeddings_list.append(target_embedding)

    time.sleep(
        SLEEP_SECONDS_BETWEEN_QUERIES
    )  # Avoid getting rate limit errors from get_embedding()

source_embeddings = torch.cat(source_embeddings_list, 0)
target_embeddings = torch.cat(target_embeddings_list, 0)


recall_at_1, recall_at_5, mean_reciprocal_rank = retrieval_eval(
    source_embeddings, target_embeddings
)

print(f"R@1: {recall_at_1}, R@5: {recall_at_5}, MRR: {mean_reciprocal_rank}, ")


"""
R@1: , R@5: , MRR: 
"""
