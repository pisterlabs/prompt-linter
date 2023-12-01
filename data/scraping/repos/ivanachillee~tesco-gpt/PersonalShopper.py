import numpy as np
import pandas as pd
from InstructorEmbedding import INSTRUCTOR
from sklearn.metrics.pairwise import cosine_similarity
import re
import openai
from decouple import config
import json

class PersonalShopper:
    def __init__(self,openai_api_key,tesco_controller_instance,):
        print("Starting Personal Shopper...")
        #Initialise things
        openai.api_key = openai_api_key
        self.tesco = tesco_controller_instance
        self.embeddings_model = INSTRUCTOR('hkunlp/instructor-large')

        #Load product lists and embeddings
        embeddings = np.load('data/embeddings.npy')
        self.products = pd.read_csv('data/tesco_groceries.csv')
        self.products['embeddings'] = embeddings.tolist()

        self.basket = []
        self.grandtotal = 0
        
        self.system_prompt = """
        You are a helpful personal shopping assistant. You'll think through, reason, search and iterate based on the user's input to find the best products for them based on their preferences and budget.
        You will ALWAYS answer exclusively in the following format:
        {"reasoning": "The reasoning behind your answer","action": "The action you'll take based on your reasoning, you can choose from the following: search, add_to_basket","action_params": "The parameters for the action you'll take (e.g. the product url and quantity for add_to_basket or the search query for search)")"}
        You will only search for specific products based on your reasoning (e.g. meat, eggs, etc. don't do general queries) and the user's goal, and you will only add one at a time to the basket. When searching you will pass the action_params as so: 'search_query'
        When doing an add_to_basket action, you will pass the action_params as so: [product_name,product_url,quantity_to_buy]
        If you can't find exactly what you're looking for, just go for the most sensible similar alternative.
        """

        self.main_prompt = input("Enter your general goal for shopping: ")
        self.budget = int(input("Enter your budget (€): "))
        
        self.thread = [{"role": "user", "content": f"This is my goal: {self.main_prompt}. My budget is: {self.budget}"}]
    
    def search_product(self,prompt,topn=15):
        """ Searches the database for products based on the natural language prompt. """
        query_embedding = self.embeddings_model.encode([["Represent the Grocery question for retrieving supporting titles: ",prompt]])
        self.products['similarity'] = cosine_similarity(self.products['embeddings'].values.tolist(),query_embedding)
        self.products.sort_values(by=['similarity'],ascending=False,inplace=True)
        results = self.products[['title','price','url']].head(topn)
        self.products.drop(columns=['similarity'],inplace=True)
        return results
        

        self.products.sort_values(by=['similarity'], ascending=False, inplace=True)
        return self.products[['title','price','url']].head(topn)

    def think(self, gpt_model="gpt-4",verbose=True):
        """ Reasons based on, and appends to the thread. """
        completion = openai.ChatCompletion.create(
        model=gpt_model,
        messages=[{"role": "system", "content": self.system_prompt}]+self.thread
        )

        reasoning_result = completion.choices[0]["message"]["content"]

        if verbose:
            print("=========== Personal Shopper ===========") 
            print(reasoning_result)
            print("========================================")
            print("")
        
        self.thread.append({"role": "assistant", "content": reasoning_result})

        action_to_take = re.search(r'"action": "(.*?)"', reasoning_result).group(1)

        if action_to_take == "search":
            search_query = re.search(r'"action_params": "(.*?)"', reasoning_result).group(1)
            search_result = self.search_product(search_query)
            return_message = f"Here are the top 15 results for your search: {search_result}"
        
        elif action_to_take == "add_to_basket":
            action_parameters = re.search(r'"action_params": (\[.*?\])', reasoning_result).group(1)
            action_parameters = json.loads(action_parameters)
            item_name,item_url,item_quantity = action_parameters
            self.tesco.add_to_basket(item_url,item_quantity)
            self.basket.append(f"{item_quantity}x of {item_name}")
            self.grandtotal += self.products[self.products['url'] == item_url]['price'].values[0] * item_quantity
            return_message = f"I've added {item_quantity}x of {item_name} to the basket. These are all the products in your basket: {self.basket}. The grand total so far is {self.grandtotal}."

        if verbose:
            print("----------- Action return message -----------")
            print(return_message)
            print("---------------------------------------------")
            print("")

        self.thread.append({"role": "user", "content": return_message})
    
    def start_shopping(self):
        """ Starts the shopping session. """
        
        self.tesco.empty_basket()

        #Reasoning loop
        while self.grandtotal < self.budget:
            self.think()