from flask import Flask, request, jsonify
import openai
import numpy as np
from numpy.linalg import norm
import pandas as pd
from fuzzywuzzy import fuzz
from scipy.spatial.distance import cdist
import csv
import ast
import json
import time

with open('secret_key.txt', 'r') as f:
    secret_key = f.read().strip()

openai.api_key = secret_key

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def cosine_similarity(v1, words):
    # Compute cosine similarity between the embeddings

    for i in range(len(words)) :
        v2 = mental_health_embeddings_dict[mental_health_words[i]]
        v2 = ast.literal_eval(v2)
        cosine_sim = np.dot(v1,v2)/(norm(v1)*norm(v2))
        print(cosine_sim)
        if(cosine_sim > 0.8):
            return True, mental_health_words[i]
    return False, "none of the topics"

messages = [
    {"role": "system", "content": "You are a helpful and kind AI Assistant on mental health topics"},
]

#### One time activity to save embeddings ####

mental_health_words = ['Relationships','Health mental','Life Choices','Career pressure','Work pressure', 'Loss/Grief',
                       'Family/Friends problems','Self Growth','Kids/Parenting problems',
    'anxiety', 'depression', 'stress', 'panic', 'trauma', 'therapy', 
                       'medication', 'psychologist', 'psychiatrist', 'counseling', 'mental illness', 
                       'bipolar', 'suicide', 'obsessive-compulsive disorder', 'PTSD', 'phobia', 'addiction', 
                       'compulsive behavior', 'schizophrenia', 'mania', 'delusion', 'paranoia', 'psychosis', 
                       'personality disorder', 'borderline personality disorder', 'narcissism', 
                       'dissociative identity disorder', 'eating disorder', 'anorexia', 'bulimia', 
                       'binge eating disorder', 'body dysmorphic disorder', 'psychosomatic disorder', 
                      'conversion disorder', 'somatoform disorder', 'hypochondria', 'psychogenic amnesia',
                         'memory loss', 'dementia', 'Alzheimer’s disease', 'obsession', 'insomnia', 
                         'sleep disorder', 'nightmare', 'sleep apnea', 'restless legs syndrome', 
                         'antisocial personality disorder', 'narcissistic personality disorder', 
                         'borderline personality disorder', 'schizoid personality disorder', 
                         'schizotypal personality disorder', 'avoidant personality disorder', 
                         'dependent personality disorder', 'obsessive-compulsive personality disorder', 
                         'paranoid personality disorder', 'depersonalization disorder', 
                         'derealization disorder', 'manic episode', 'hypomanic episode', 'mixed episode',
                           'major depressive episode', 'hallucination', 'delusion', 'suicidal ideation',
                             'mania', 'psychotic episode', 'postpartum depression', 'perinatal depression', 
                             'premenstrual dysphoric disorder', 'gender dysphoria', 'gender identity disorder',
                              'identity crisis', 'body image', 'self-esteem', 'self-harm', 'self-injury', 
                               'suicide prevention', 'anger management', 'cognitive behavioral therapy', 
                               'dialectical behavior therapy', 'exposure therapy', 'interpersonal therapy',
                                 'psychodynamic therapy', 'group therapy', 'family therapy', 'support groups', 
                                 'mindfulness', 'meditation', 'breathing exercises', 'yoga', 'exercise',
                                   'nutrition', 'self-care', 'relaxation techniques']

get_embebedding_again = True
if(get_embebedding_again):

    mental_health_embeddings_dict = {}

    counter = 1
    for i, j  in enumerate(mental_health_words):
        if(counter % 20 == 0):
            time.sleep(20) #For adhering to request limit 
        counter += 1
        #print(counter)
        mental_health_embeddings_dict[mental_health_words[i]] = get_embedding(j) 

    with open('data/mental_health_embeddings_dict.json', 'w') as fp:
        json.dump(mental_health_embeddings_dict, fp)             

    with open("data/embeddings_mental_health_words.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in mental_health_embeddings_dict.items():
            writer.writerow([key, value])

################################ 

# open the CSV file for reading
with open('data/embeddings_mental_health_words.csv', mode='r') as file:

    # create a csv reader object
    reader = csv.reader(file)

    # create an empty dictionary
    mental_health_embeddings_dict = {}

    # iterate over each row in the CSV file
    for row in reader:

        # get the key and value from the row
        key = row[0]
        value = row[1]

        # add the key-value pair to the dictionary
        mental_health_embeddings_dict[key] = value

with open('mental_health_embeddings_dict.json', 'w') as f:
    json.dump(mental_health_embeddings_dict, f)
 


app = Flask(__name__)

# Define a secret key to be used for authentication
with open('app_secret_key.txt', 'r') as f:
    secret_key = f.read().strip()

#print(secret_key)

@app.route('/chat', methods=['POST'])
def prompt():
    #auth_header = request.headers.get('Authorization')
    #if auth_header != f'Token {secret_key}':
        #return 'Unauthorized', 401
    
    data = request.get_json()
    if 'prompt' not in data or 'openai_key' not in data:
        return 'Missing input data', 400
    if(not isinstance(data['prompt'], str)):
       return 'Prompt input must be a string', 400
    
    
    prompt = data['prompt']
    openai_key = data['openai_key']

    openai.api_key = openai_key

    sim, topic = cosine_similarity(get_embedding(prompt), mental_health_words)  
    print(sim, topic)
    if(not sim):
        reply = "Sorry, I am AI bot to assist on the specific topic of mental health"
    else:
        messages.append({"role": "user", "content": prompt})
        chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
            )

        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
           
    return reply, 200

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host= '0.0.0.0', port=5000)
