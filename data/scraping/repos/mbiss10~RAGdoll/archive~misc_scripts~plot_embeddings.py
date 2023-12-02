from nomic import atlas
import numpy as np
import pickle
import os
import openai

from openai.embeddings_utils import cosine_similarity

v1 = np.random.randint(1000, size=500).tolist()
v2 = np.random.randint(1000, size=500).tolist()
sim = cosine_similarity(v1, v2)
print(sim)

# from dotenv import load_dotenv
# load_dotenv()
# API_KEY = os.getenv('OPENAI_API_KEY')
# os.environ["OPENAI_API_KEY"] = API_KEY

# cs_data = None
# with open("./data/dev/cs_data.pkl") as f:
#     cs_data = pickle.load(f)



# embedding = openai.Embedding.create(
#     input="Your text goes here", model="text-embedding-ada-002"
# )["data"][0]["embedding"]

# project = atlas.map_embeddings(embeddings=embeddings)

# from datasets import load_dataset
# dataset = load_dataset('ag_news')['train']

# max_documents = 10000
# subset_idxs = np.random.randint(len(dataset), size=max_documents).tolist()
# documents = [dataset[i] for i in subset_idxs]
# print(documents[0:10])

# project = atlas.map_text(data=documents,
#                           indexed_field='text',
#                           name='News 10k Example',
#                           colorable_fields=['label'],
#                           description='News 10k Example.'
#                           )


# db = None
# with open("./data/dev/db_cs_with_sources.pkl", "rb") as f:
#     db = pickle.load(f)

# print(db.docstore)
# print(db.index)