import base64
import json

import torch
from numpy.linalg import norm
from openai import OpenAI
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForSequenceClassification,
                          AutoTokenizer)

client = OpenAI()

with open("real.jpg", "rb") as f:
    encoded_image = base64.b64encode(f.read())

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful ingredient identifier."
        },
        {
            "role": "user",
            "content": [
                "Output a json list of edible ingredients in the image. You must ONLY output a json list, not a json object, so that it is easily parsable by a program. You cannot use markdown formatting.",
                {"image": encoded_image.decode("utf-8")},
            ],
        }
    ],
    max_tokens=1024,
)

ingredients = json.loads(response.choices[0].message.content)

print(ingredients)

# cos_sim = lambda a, b: (a @ b.T) / (norm(a)*norm(b))
embeddings_model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
embedding = embeddings_model.encode("\n".join(ingredients))

print(embedding)

# print(cos_sim(embeddings[0], embeddings[2]))

# print(embeddings[0].shape)