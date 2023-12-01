import openai
import json # for data storage and retrieval in sqlite
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) # for exponential backoff

wordcount_limit  = 300 # 300 English words

client = openai.OpenAI()

''''NEEDS API COMMUNICATION: CREATING A PERSON'''
'''RECORD EACH PERSON'S CREATION TIME, WE HAVE RATE LIMITS OF 3 REQUEST PER MINUTE AND 200 REQUESTS PER DAY'''

@retry(wait=wait_random_exponential(min=25, max=500), stop=stop_after_attempt(8))
def embeddings_with_backoff(input_text):
    return client.embeddings.create(
        input = input_text,
        model = "text-embedding-ada-002" # text-embedding-ada-002
    )   

class Person:
    def __init__(self, name: str, age: int, man: bool, hetero: bool, city: str, email: str, expectation: str):
        self.name = name
        self.id = name + "#" + email # generate a unique id for each person
        self.age = age
        self.man = man # if user is a man
        self.hetero = hetero # if user is heterosexual
        self.city = city
        self.email = email
        self.expectation = expectation # expectation from the person
        self.wordcount = self.expectation.count(' ') + 1
        self.match_result_id = [] # empty by default
        self.token_cost = 0 # 0 by default
        self.vector = [] # empty by default
        
        # generating the vector for expectation
        if self.wordcount <= wordcount_limit:
            # response = client.embeddings.create(
            #     input = self.expectation,
            #     model = "text-embedding-ada-002"
            # )
            try:
                response = embeddings_with_backoff(self.expectation)
                self.token_cost = response.usage.total_tokens
                self.vector = response.data[0].embedding # a list with 1536 elements
            except openai.OpenAIError as e:
                print(f"Error: {e}")
                '''API COMMUNICATIONS HERE'''
        else:
            ''''NEEDS API COMMUNICATION'''
            print("Word count limit exceeded, please try again")
            raise ValueError(f"Word count limit exceeded. Limit: {wordcount_limit} words.")
            
        ''''NEEDS API COMMUNICATION'''
        print(f"{self.id} created successfully")
    
    def to_dict(self):
        
        return {
            'name': self.name,
            'id': self.id,
            'age': self.age,
            'man': self.man,
            'hetero': self.hetero,
            'city': self.city,
            'email': self.email,
            'expectation': self.expectation,
            'wordcount': self.wordcount,
            # Serialize list and dict data to JSON for storage
            'match_result_id': json.dumps(self.match_result_id),
            'token_cost': self.token_cost,
            'vector': json.dumps(self.vector)
        }
        
    def __str__(self):
        return f"{self.name} has ID {self.id}."