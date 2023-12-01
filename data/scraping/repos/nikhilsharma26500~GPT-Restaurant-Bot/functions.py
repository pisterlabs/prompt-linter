# Using functions.py to make queries from the databases to retrieve correct information for the bot to respond to the user.

import os

# Pizza, Order, Review classes are defined in the db.py file in the app package which is why app.db is used. Importing Session object as well.
# from app.db import Session, Pizza, Order, Review
# from app.prompts import QA_PROMPT
from db import Session, Pizza, Order, Review
from prompts import QA_PROMPT
import json
from langchain.llms import OpenAI

'''
The RetrievalQA class is a question-answering model that uses a retrieval-based approach to answer questions. It works by first retrieving a set of candidate answers from a large corpus of text, and then ranking the candidates based on their relevance to the question.
'''
from langchain.chains import RetrievalQA
# from app.store import get_vectorstore
from store import get_vectorstore


'''
This function is used to retrieve information about a specific pizza from the database.
'''
def get_pizza_info(pizza_name: str):
    session = Session() # Creates a new session and gives access to the database.
    '''
    The query() method on the session object is used to query the database.
    The query() method takes a model class (here Pizza) as an argument and returns a query object that can be used to query the database.
    The filter() method on the query object is used to filter the results of the query in the name column.
    The we extract the first result from the query using the first() method.
    '''
    pizza = session.query(Pizza).filter(Pizza.name == pizza_name).first()
    session.close()
    if pizza:
        return json.dumps(pizza.to_json()) # Returns a JSON representation of the pizza object.
    else:
        return "Pizza not found"
    
    
    
'''
This function is used to retrieve information about a specific order from the database.
'''  
def create_order(pizza_name: str):
    session = Session()
    pizza = session.query(Pizza).filter(Pizza.name == pizza_name).first()
    if pizza:
        order = Order(pizza=pizza) # Creates a new Order instance and sets the pizza attribute to the pizza object.
        session.add(order) # Adds the order to the session.
        session.commit() # Commits the changes to the database.
        session.close()
        return "Order created"
    else:
        session.close()
        return "Pizza not found"
    

'''
This function is used to retrieve information about a specific order from the database.
'''
def create_review(review_text: str):
    session = Session()
    review = Review(review=review_text)
    session.add(review)
    session.commit()
    session.close()
    return "Review created"



'''
This function talks to the vector database
'''
def ask_vector_db(question: str):
    llm = OpenAI(openai_api_key=os.environ('OPENAI_API_KEY')) # RetruevalQA class requires a language model to be passed to it.
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        # The get_vectorstore() function is called to retrieve the vector store
        # The as_retriever() method is then called on the vector store object to create a retriever object.
        retriever = get_vectorstore().as_retriever(),
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    
    result = qa.run(question)
    return result


'''
This dictionary is used to provide an API for interacting with the application.
By mapping function names to function objects, the API can be used to call these functions remotely.
The LLM will use these keys to call the respective functions.
'''
api_functions = {
    "create_review": create_review,
    "create_order": create_order,
    "ask_vector_db": ask_vector_db,
    "get_pizza_info": get_pizza_info,
}


# Initialising the Pizza in the database
def create_pizzas():
    session = Session()
    
    # Dictionary of pizza names and prices
    pizzas = {
        "Margherita": 7.99,
        "Pepperoni": 8.99,
        "BBQ Chicken": 9.99,
        "Hawaiian": 8.49,
        "Vegetarian": 7.99,
        "Buffalo": 9.49,
        "Supreme": 10.99,
        "Meat Lovers": 11.99,
        "Taco": 9.99,
        "Seafood": 12.99,
    }
    
    # Loop through the pizzas dictionary and create a Pizza object for each pizza.
    for name, price in pizzas.items():
        pizza = Pizza(name=name, price=price) # Creates a new Pizza instance.
        session.add(pizza) # Adds the pizza to the session.
        
    session.commit() # Commits the changes to the database.
    session.close()