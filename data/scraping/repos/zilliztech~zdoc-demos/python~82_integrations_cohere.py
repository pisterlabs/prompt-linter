from pymilvus import connections, DataType, CollectionSchema, FieldSchema, Collection, utility
import cohere
import pandas
import numpy as np
from tqdm import tqdm
import time, os

# Set up arguments

# 1. Set the The SQuAD dataset url.
FILE = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json' 

# 2. Set up the name of the collection to be created.
COLLECTION_NAME = 'question_answering_db'

# 3. Set up the dimension of the embeddings.
DIMENSION = 1024

# 4. Set the number of entities to create and the number of entities to insert at a time.
COUNT = 5000
BATCH_SIZE = 96

# 5. Set up the cohere api key
COHERE_API_KEY = "YOUR_COHERE_API_KEY"

# 6. Set up the connection parameters for your Zilliz Cloud cluster.
URI = 'YOUR_CLUSTER_ENDPOINT'

# 7. Set up the token for your Zilliz Cloud cluster.
# You can either use an API key or a set of cluster username and password joined by a colon.
TOKEN = 'YOUR_CLUSTER_TOKEN'

# Download the dataset
dataset = pandas.read_json(FILE)

# Clean up the dataset by grabbing all the question answer pairs
simplified_records = []
for x in dataset['data']:
    for y in x['paragraphs']:
        for z in y['qas']:
            if len(z['answers']) != 0:
                simplified_records.append({'question': z['question'], 'answer': z['answers'][0]['text']})

# Grab the amount of records based on COUNT
simplified_records = pandas.DataFrame.from_records(simplified_records)
simplified_records = simplified_records.sample(n=min(COUNT, len(simplified_records)), random_state = 42)

# Check if the length of the cleaned dataset matches COUNT
print(len(simplified_records))

# Output
#
# 5000



# Connect to Zilliz Cloud and create a collection

connections.connect(
    alias='default',
    # Public endpoint obtained from Zilliz Cloud
    uri=URI,
    token=TOKEN
)

if COLLECTION_NAME in utility.list_collections():
    utility.drop_collection(COLLECTION_NAME)

fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='original_question', dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name='answer', dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name='original_question_embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
]

schema = CollectionSchema(fields=fields)

collection = Collection(
    name=COLLECTION_NAME,
    schema=schema,
)

index_params = {
    'metric_type': 'IP',
    'index_type': 'AUTOINDEX',
    'params': {'nlist': 1024}
}

collection.create_index(
    field_name='original_question_embedding', 
    index_params=index_params
)

collection.load()

# Set up a Cohere client
cohere_client = cohere.Client(COHERE_API_KEY)

# Extract embeddings from questions using Cohere
def embed(texts, input_type):
    res = cohere_client.embed(texts, model='embed-multilingual-v3.0', input_type=input_type)
    return res.embeddings

# Insert each question, answer, and qustion embedding
total = pandas.DataFrame()
for batch in tqdm(np.array_split(simplified_records, (COUNT/BATCH_SIZE) + 1)):
    questions = batch['question'].tolist()
    embeddings = embed(questions, "search_document")
    
    data = [
        {
            'original_question': x,
            'answer': batch['answer'].tolist()[i],
            'original_question_embedding': embeddings[i]
        } for i, x in enumerate(questions)
    ]

    collection.insert(data=data)

time.sleep(10)

# Search the cluster for an answer to a question text
def search(text, top_k = 5):

    # AUTOINDEX does not require any search params 
    search_params = {}

    results = collection.search(
        data = embed([text], "search_query"),  # Embeded the question
        anns_field='original_question_embedding',
        param=search_params,
        limit = top_k,  # Limit to top_k results per search
        output_fields=['original_question', 'answer']  # Include the original question and answer in the result
    )

    distances = results[0].distances
    entities = [ x.entity.to_dict()['entity'] for x in results[0] ]

    ret = [ {
        "answer": x[1]["answer"],
        "distance": x[0],
        "original_question": x[1]['original_question']
    } for x in zip(distances, entities)]

    return ret
            

# Ask these questions
search_questions = ['What kills bacteria?', 'What\'s the biggest dog?']

# Print out the results in order of [answer, similarity score, original question]

ret = [ { "question": x, "candidates": search(x) } for x in search_questions ]

print(ret)

# Output
#
# [
#     {
#         "question": "What kills bacteria?",
#         "candidates": [
#             {
#                 "answer": "farming",
#                 "distance": 0.6261022090911865,
#                 "original_question": "What makes bacteria resistant to antibiotic treatment?"
#             },
#             {
#                 "answer": "Phage therapy",
#                 "distance": 0.6093736886978149,
#                 "original_question": "What has been talked about to treat resistant bacteria?"
#             },
#             {
#                 "answer": "oral contraceptives",
#                 "distance": 0.5902313590049744,
#                 "original_question": "In therapy, what does the antibacterial interact with?"
#             },
#             {
#                 "answer": "slowing down the multiplication of bacteria or killing the bacteria",
#                 "distance": 0.5874154567718506,
#                 "original_question": "How do antibiotics work?"
#             },
#             {
#                 "answer": "in intensive farming to promote animal growth",
#                 "distance": 0.5667208433151245,
#                 "original_question": "Besides in treating human disease where else are antibiotics used?"
#             }
#         ]
#     },
#     {
#         "question": "What's the biggest dog?",
#         "candidates": [
#             {
#                 "answer": "English Mastiff",
#                 "distance": 0.7875324487686157,
#                 "original_question": "What breed was the largest dog known to have lived?"
#             },
#             {
#                 "answer": "forest elephants",
#                 "distance": 0.5886962413787842,
#                 "original_question": "What large animals reside in the national park?"
#             },
#             {
#                 "answer": "Rico",
#                 "distance": 0.5634892582893372,
#                 "original_question": "What is the name of the dog that could ID over 200 things?"
#             },
#             {
#                 "answer": "Iditarod Trail Sled Dog Race",
#                 "distance": 0.546872615814209,
#                 "original_question": "Which dog-sled race in Alaska is the most famous?"
#             },
#             {
#                 "answer": "part of the family",
#                 "distance": 0.5387814044952393,
#                 "original_question": "Most people today describe their dogs as what?"
#             }
#         ]
#     }
# ]


    