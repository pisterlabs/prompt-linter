import os
import json
import dotenv
import numpy as np

import openai
import weaviate

dotenv.load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
# os.environ['WEAVIATE_API_KEY']
# os.environ['WEAVIATE_API_URL']
# os.environ['OPENAI_API_KEY']
# os.environ['WEAVIATE_API_URL'] + '/v1/objects'


hf_openai_embedding = lambda x: np.array(openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'], dtype=np.float64)

# X-Cohere-Api-Key
# X-HuggingFace-Api-Key
tmp0 = weaviate.auth.AuthApiKey(os.environ['WEAVIATE_API_KEY'])
tmp1 = {"X-OpenAI-Api-Key": os.environ['OPENAI_API_KEY']} #optional
client = weaviate.Client(url=os.environ['WEAVIATE_API_URL'], auth_client_secret=tmp0, additional_headers=tmp1)

# vectorizer: text2vec-openai text2vec-cohere text2vec-huggingface
client.schema.create_class({"class": "Question", "vectorizer": "text2vec-openai"})
client.schema.get()
# client.schema.delete_class("Question")

# wget https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json
with open('data/jeopardy_tiny.json', 'r', encoding='utf-8') as fid:
    data = json.load(fid)
with client.batch as batch:
    batch.batch_size=100
    for x in data:
        tmp0 = dict(answer=x["Answer"], question=x["Question"], category=x["Category"])
        client.batch.add_data_object(tmp0, "Question")


result = client.data_object.get() #vector not included

nearText = {"concepts": ["biology"]}
result = client.query.get("Question", ["question", "answer", "category"]).with_near_text(nearText).with_limit(2).do()
result['data']['Get']['Question'] #(list,(dict),2)
'''
{
    "answer": "DNA",
    "category": "SCIENCE",
    "question": "In 1953 Watson & Crick built a model of the molecular structure of this, the gene-carrying substance"
},
{
    "answer": "species",
    "category": "SCIENCE",
    "question": "2000 news: the Gunnison sage grouse isn't just another northern sage grouse, but a new one of this classification"
}
'''

tmp0 = {"vector": hf_openai_embedding('biology'), "certainty": 0.7}
result = client.query.get("Question", ["question", "answer"]).with_near_vector(tmp0).with_limit(2).do()
result = client.query.get("Question", ["question", "answer"]).with_near_vector(tmp0).with_limit(2).with_additional(['certainty']).do()
result['data']['Get']['Question'][0]['_additional']['certainty'] #0.9015896916389465
result = client.query.get("Question", ["question", "answer"]).with_near_vector(tmp0).with_limit(1).with_additional(['vector']).do()
np0 = np.array(result['data']['Get']['Question'][0]['_additional']['vector'], dtype=np.float64) #(np,float64,1536) norm(np0)=1


tmp0 = {"path": ["category"], "operator": "Equal", "valueText": "ANIMALS"}
result = client.query.get("Question", ["question", "answer", "category"]).with_where(tmp0).do()

result = client.query.aggregate("Question").with_fields("meta {count}").do()

tmp0 = {"path": ["category"], "operator": "Equal", "valueText": "ANIMALS"}
result = client.query.aggregate("Question").with_where(tmp0).with_fields("meta {count}").do()

class_schema = {
    "class": "Question",
    "description": "Information from a Jeopardy! question",
    "properties": [
        {"name": "question", "dataType": ["text"], "description": "question-description"},
        {"name": "answer", "dataType": ["text"], "description": "answer-description"},
        {"name": "category", "dataType": ["text"], "description": "category-description"},
    ],
    "vectorizer": "text2vec-openai",
}
