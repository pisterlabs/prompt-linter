from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pymongo import MongoClient
from pymongo.server_api import ServerApi

import os
from dotenv import load_dotenv

import random
# importing nlp dependencies
import cohere
import cohere.classify as co_classify
import os
import numpy as np
import pandas as pd

from datetime import date
from bson.json_util import dumps, loads

from pydantic import BaseModel

origins = [
    "http://localhost:3000",

    "http://localhost:3000/entries",
    "http://127.0.0.1:8000",
]


load_dotenv()

# fetching api key for cohere
co_client = cohere.Client(f'{os.getenv("COHERE_KEY")}')

# using fast api
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

password = os.environ.get('PASSWORD')

client = MongoClient(f'mongodb+srv://voicejournalhtn22:{password}@cluster0.0wr1fib.mongodb.net/?retryWrites=true&w=majority',)
db = client.test
collection = db.htn2022
collection2 = db.testing


@app.get("/upload")
def read_root():
    x = collection2.insert_one({"Hello": "World"}).inserted_id

    return {"Hello": "World"}



# root dir
@app.get("/")
async def read_root():
    return {
        "Running on this localhost"
    }

class Data(BaseModel):
    transcript: str
    summary : str
# analyzes the text to determine the mood from the text
# prod: vector for [happy transcript, sad transcript] 
@app.post('/sheesh')
async def analyze_transcript(data: Data):
    # happiness vector
    day_decoding = ['miserable', 'sad', 'neutral', 'happy', 'ecstatic']

    # classifying the transcript
    try:
        response = co_client.classify(
            inputs = [f"{data.transcript}"],
            examples = [
                co_classify.Example('My grandmother was tragically hit by a car yesterday.', 'miserable'),
                co_classify.Example("I'm feeling so depressed I think I'm going to kill myself.", 'miserable'),
                co_classify.Example("My girlfriend has been cheating on me for the past 5 years.", 'miserable'),
                co_classify.Example("I feel lonely, I hate isolation.", 'miserable'),
                co_classify.Example("My doctor told me I only have 1 week to live.", 'miserable'),
                co_classify.Example('I dropped my food on the ground by accident.', 'sad'),
                co_classify.Example('I lost my sock this morning', 'sad'),
                co_classify.Example('I cracked the screen of my phone.', 'sad'),
                co_classify.Example('My friend accidentally sat on my laptop', 'sad'),
                co_classify.Example('I stubbed my toe on my bed', 'sad'),
                co_classify.Example('I ate an apple today.', 'neutral'),
                co_classify.Example('She put the wrapper in the garbage.', 'neutral'),
                co_classify.Example('I washed my shirt before going to the mall.', 'neutral'),
                co_classify.Example('He got out of the shower at 10:00 pm.', 'neutral'),
                co_classify.Example('Jerry peeled an orange.', 'neutral'),
                co_classify.Example('The party was super fun.', 'happy'),
                co_classify.Example('We were able to create a successful project', 'happy'),
                co_classify.Example('I aced all my quizzes.', 'happy'),
                co_classify.Example('My friends are super caring and motivate me to be the best.', 'happy'),
                co_classify.Example('Today was a great day.', 'happy'),
                co_classify.Example('I won the lottery three times in a row.', 'ecstatic'),
                co_classify.Example('I found the love of my life.', 'ecstatic'),
                co_classify.Example('I achieved all my goals and dreams.', 'ecstatic'),
                co_classify.Example('I am living my best life and love myself.', 'ecstatic'),
                co_classify.Example('I have a wonderful family and amazing friends.', 'ecstatic'),
            ]
    )
    except:
        print("Error with Cohere")        
    # vector encoding of happy and sad
    
    try:
        response_labels = response.classifications[0].labels
        happiness_encoding = ([
            response_labels['miserable'].confidence, 
            response_labels['sad'].confidence, 
            response_labels['neutral'].confidence, 
            response_labels['happy'].confidence, 
            response_labels['ecstatic'].confidence
        ])
    except:
        print('error w cohere 2')

    prompt = f'''"At the park, there was a murderer at night and I almost was not able to escape with my life and as a result, I am traumatised"
In summary: "The park is dangerous at night"

"At the carnival, there was a variety of food to pick from and I was never happier"
In summary:"Lot's of food makes me happy"

{data.transcript}
In summary:"'''
    n_generations = 5

    prediction = co_client.generate(
        model = 'large', 
        prompt = prompt, 
        return_likelihoods = 'GENERATION', 
        stop_sequences = ['"'], 
        max_tokens = 50,
        temperature = 0.7, 
        num_generations = n_generations, 
        k = 0,
        p = 0.75
    )

    gens = []
    likelihoods = []
    for gen in prediction.generations:
        gens.append(gen.text)

        sum_likelihood = 0
        for t in gen.token_likelihoods:
            sum_likelihood += t.likelihood
        # Get sum of likelihoods
        likelihoods.append(sum_likelihood)

    pd.options.display.max_colwidth = 200
    # Create a dataframe for the generated sentences and their likelihood scores
    df = pd.DataFrame({'generation':gens, 'likelihood': likelihoods})
    # Drop duplicates
    df = df.drop_duplicates(subset=['generation'])
    # Sort by highest sum likelihood
    df = df.sort_values('likelihood', ascending=False, ignore_index=True)

    today = date.today()

    # // NOTE: this information will be added to the database
    result = { 
        '_id' : str(random.randint(0,90000000)),
        'speech': data.transcript,
        'summary' : df.generation[0].replace("\"", ""),
    
        'rating': int(np.argmax(happiness_encoding)),
        'date' : str(today)
         }
    
    x = collection.insert_one(result)
    return result


# summarizes the text
# NOTE: NEED TO FIX THIS SHT
@app.post('/summarize_transcript/')
async def summarize__transcript(transcript : str):
    prompt = f'''"At the park, there was a murderer at night and I almost was not able to escape with my life and as a result, I am traumatised"
In summary: "The park is dangerous at night"

"At the carnival, there was a variety of food to pick from and I was never happier"
In summary:"Lot's of food makes me happy"

{transcript}
In summary:"'''
    n_generations = 5

    prediction = co_client.generate(
        model = 'large', 
        prompt = prompt, 
        return_likelihoods = 'GENERATION', 
        stop_sequences = ['"'], 
        max_tokens = 50,
        temperature = 0.7, 
        num_generations = n_generations, 
        k = 0,
        p = 0.75
    )

    gens = []
    likelihoods = []
    for gen in prediction.generations:
        gens.append(gen.text)

        sum_likelihood = 0
        for t in gen.token_likelihoods:
            sum_likelihood += t.likelihood
        # Get sum of likelihoods
        likelihoods.append(sum_likelihood)

    pd.options.display.max_colwidth = 200
    # Create a dataframe for the generated sentences and their likelihood scores
    df = pd.DataFrame({'generation':gens, 'likelihood': likelihoods})
    # Drop duplicates
    df = df.drop_duplicates(subset=['generation'])
    # Sort by highest sum likelihood
    df = df.sort_values('likelihood', ascending=False, ignore_index=True)

    result = df['generation'][0]
    result = result[0 : max(0, len(result) - 2)]

    return result


@app.get("/letsgetthisbread")
async def read_root():
    info = list(collection.find({}))
    
    return info

