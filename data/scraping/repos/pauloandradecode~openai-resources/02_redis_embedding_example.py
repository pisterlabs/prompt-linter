import redis
import openai
import getpass
import time
import numpy as np
import json


# Varianles globales
openai.api_key = getpass.getpass()
model = 'text-embedding-ada-002'

# creamos un cliente de redis
class RedisClient:
    def __init__(self, host='localhost', port=6379):
        try:
            self.client = redis.Redis(host=host, port=port)
        except ConnectionError:
            print("Connection Error")

    def set(self, key, value):
        try:
            return self.client.set(key, json.dumps(value))
        except ConnectionError:
            print("Connection Error")

    def get(self, key):
        try:
            return json.loads(self.client.get(key))
        except ConnectionError:
            print("Connection Error")

    def delete(self, key):
        try:
            print("Deleting key: " + key)
            return self.client.delete(key)
        except ConnectionError:
            print("Connection Error")

    def keys(self, pattern='*'):
        try:
            return self.client.keys(pattern)
        except ConnectionError:
            print("Connection Error")

    
    def exists(self, key):
        try:
            return self.client.exists(key)
        except ConnectionError:
            print("Connection Error")
            

    def flushall(self):
        try:
            return self.client.flushall()
        except ConnectionError:
            print("Connection Error")

    def flushdb(self):
        try:
            return self.client.flushdb()
        except ConnectionError:
            print("Connection Error")


# create a redis client
redis_client = RedisClient()

# retrieve the embedding from the redis cache
# if it doesn't exist, create it and store it in the cache
# then return it

def get_embedding(text):
    # redis_client.delete(text)

    # if the embedding doesn't exist in the cache
    if not redis_client.exists(text):
        # create the embedding
        time.sleep(1)
        print('Creating embedding for: ' + text)
        
        response = openai.Embedding.create(
            input = text,
            model = model
        )

        # print(response['data'][0]['embedding'])
        embeddings = response['data'][0]['embedding']
        
        # store the embedding in the cache
        redis_client.set(text, embeddings)
    # return the embedding
    else:
        embeddings = redis_client.get(text)
    return embeddings


# calculate the cosine similarity between two embeddings
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# calculate the similarity between two texts
def similarity(text1, text2):
    # get the embeddings for the texts
    embeddings1 = get_embedding(text1)
    embeddings2 = get_embedding(text2)

    # calculate the cosine similarity between the embeddings
    return cosine_similarity(embeddings1, embeddings2)


# calculate the similarity between a text and a list of texts
def similarity_list(text, text_list):
    # get the embedding for the text
    embeddings = get_embedding(text)
    
    # calculate the cosine similarity between the embedding and each text in the list
    return [cosine_similarity(embeddings, get_embedding(text2)) for text2 in text_list]


# Calculate the similarity between a text and a list of texts and return the most similar text
def most_similar(text, text_list):
    
    # get the list of similarities
    similarities = similarity_list(text, text_list)
    
    # return the text with the highest similarity
    return text_list[np.argmax(similarities)]


# calculate the similarity between a text and a list of texts
# and return the most similar text and its similarity
def most_similar_with_similarity(text, text_list):
    # get the list of similarities
    similarities = similarity_list(text, text_list)

    # get the index of the highest similarity
    index = np.argmax(similarities)

    # return the text with the highest similarity and its similarity
    return text_list[index], similarities[index]


# calculate the similarity between a text and a list of texts
# and return the most similar text and its similarity
# if the similarity is below a threshold, return None
def most_similar_with_similarity_threshold(text, text_list, threshold):
    # get the list of similarities
    similarities = similarity_list(text, text_list)

    # get the index of the highest similarity
    index = np.argmax(similarities)

    # if the similarity is below the threshold
    if similarities[index] < threshold:
        # return None
        return None, None
    # otherwise
    else:
        # return the text with the highest similarity and its similarity
        return text_list[index], similarities[index]


# Ejemplo de funcionamiento
# embeddings = get_embedding("This is a test")
# print(embeddings)

# openai.Engine.list() 

# BÚSQUEDAS INTELIGENTES CON OPENAI
text_list = ['Los felinos dicen', 'Los caninos dicen', 'Los bovinos dicen', 'Tengo un terreno en ayacucho', 'Tengo una casa en Piura']

text = 'soy rico'

print(f'El texto {text} es similar a: {most_similar(text, text_list)}')
