import json
from unittest import result
from xml.dom.minidom import Document
from cohere.classify import Example
from sklearn.model_selection import train_test_split
from tokenize import String
from urllib import response
from fastapi import FastAPI
from fastapi.logger import logger
from pydantic import BaseModel
import pandas as pd
import cohere

from backend_scripts.prompt_generation import cohereExtractor, CohereClassifier
app = FastAPI()

api_key = pd.read_json("../config/apikey.json")['cohere_key'][0]
co = cohere.Client(api_key)

class EntityDocument(BaseModel):
	document: str

class NewsDocument(BaseModel):
    document: str
    document_part: str

def get_entity_extraction_prompt(number):
    
    f = open("../data/output/entity.txt", "r")
    examples = f.read()
    
    new_examples = examples.split("----")
    
    if number > 0:
        if number <= len(new_examples):
            new_examples = "".join([e+"----" for e in new_examples[:number]])
            return cohereExtractor(examples=new_examples, co=co),new_examples
        else:
            return cohereExtractor(examples=examples, co=co),examples
    else:
        return None

def get_news_scoring_prompt():
    
    df = pd.read_csv('../data/news/news_data.csv')
    df['Analyst_Rank'] = df['Analyst_Rank'].apply(lambda x: 0 if x < 4 else 1)

        
    # Split the dataset into training and test portions
    # Training = For use in Sections 2 and 3
    # Test = For evaluating the classifier performance
    X, y = df["Description"], df["Analyst_Rank"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=21)

    # Collate the examples via the Example module
    examples = list()
    for txt, lbl in zip(X_train, y_train):
        examples.append(Example(txt, lbl))
        
    return examples

@app.post('/jdentities')
def get_entities(data: EntityDocument):
    print('something')
    received = data.dict()
    print(received)
    document = received['document']
    # number = received['number']
    document = "\n"+document.replace("\n", " ") + '\n\nExtracted Text:'
    
    print(document)
    
    cohereExtractor,examples = get_entity_extraction_prompt(number=3)

    extracted_text = cohereExtractor.extract(document)
     
    response = extracted_text
    
    logger.info(response)
    
    return response

@app.post('/dnewscore')
def get_score(data: NewsDocument):

    received = data.dict()
    document = received['document']
    document_part = received['document_part']
    print(document_part)
    examples = get_news_scoring_prompt()
    
    print(document)

    cohereClassifier = CohereClassifier(examples=examples,co=co)
    
    result = cohereClassifier.classify_text(document)
    
    score,prob='',''
    
    for r in result[0].confidence:
        if result[0].prediction == r.label:
           prob = r.confidence
           score = result[0].prediction

    response = 'score '+score+' with confidence of '+str(prob)
    print(response)
    logger.info(response)

    return response

# homepage route
@app.get("/")
def read_root():
	return {'message': 'This is the homepage of the API '}
