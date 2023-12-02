import requests
import openai
import pandas as pd
import numpy as np
import pinecone
import json
import os
from dotenv import load_dotenv

load_dotenv()  

openai.api_key = os.environ.get("API_KEY")

summary_df = pd.read_csv("summary_embeddings.csv")

actions = [
    [
        "0", "scroll", "GLOBAL", 
        {
            "name": "scroll",
            "description": "Scroll the screen",
            "parameters" : {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["Up", "Down"],
                        "description": "The direction to move in the screen. Moves either up or down"
                    }
                },
                "required":["direction"]
            }
        }
    ],
    [
        "1", "search", "GLOBAL",
        {
            "name": "search",
            "description": "Search for some product in Verizon",
            "parameters" : {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "enum": ["Personal Phone", "Business Phone", "Tablets & Laptops"],
                        "description": "The product to search",
                    }
                },
                "required":["product"]
            }
        }
    ],
    [
        "2", "get_filter", "LOCAL_PER_PHONE",
        {
            "name": "get_filters",
            "description": "Parameters to filter phones",
            "parameters": {
                "type": "object",
                "properties": {
                    "brand": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["Apple", "Samsung", "Google", "Motorola", "Kyocera", "Nokia", "TCL", "Sonim"]
                        }
                    },
                    "os": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["Android", "Apple iOS"]
                        }
                    },
                    "special_offers": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["Bill Credit", "Trade In"]
                        }
                    },
                    "price": {
                        "type": "array",
                        "items": {
                            "type": "integer"
                        },
                        "description": "size 2 array with the price lower bound (index 0) and upper bound (index 1)"
                    },
                    "condition": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["New", "Certified Pre-Owned"]
                        }
                    },
                    "availability": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["Exclude Out Of Stock"]
                        }
                    },
                    "color": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["Black", "White", "Blue", "Green", "Gray", "Purple", "Red", "Pink", "Silver", "Gold", "Yellow", "Brown", "Metallic"]
                        }
                    }
                }
            }
        }
    ],
    [
        "3", "get_sort_by", "LOCAL_PER_PHONE",
        {
            "name": "get_sort_by",
            "description": "Parameters to sort the products",
            "parameters" : {
                "type": "object",
                "properties": {
                    "sort_by": {
                        "type": "string",
                        "enum": ["Featured", "Best Sellers", "Newest", "Price Low to High", "Price High to Low"]
                    },
                },
                "required":["sort_by"],
            },
        }
    ],
    [
        "4", "get_filter", "LOCAL_TABLET",
        {
            "name": "get_filter",
            "description": "Get the parameters to filter the tablets",
            "parameters": {
                "type": "object",
                "properties": {
                    "brand": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["Apple", "Samsung", "TCL", "Orbic", "Lenovo", "CTL", "RAZER"]
                        }
                    },
                    "os": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["Android", "Apple iOS", "Windows", "Chrome"]
                        }
                    },
                    "special_offers": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["Bill Credit", "Trade In"]
                        }
                    },
                    "price": {
                        "type": "array",
                        "items": {
                            "type": "integer"
                        },
                        "description": "size 2 array with the price lower bound (index 0) and upper bound (index 1)"
                    },
                "series": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["Other Tablets", "iPad Pro", "Galaxy Tab", "Chromebook", "Laptops", "iPad Air", "iPad Generation", "iPad Mini"]
                        }
                    },
                    "condition": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["New", "Certified Pre-Owned"]
                        }
                    },
                    "availability": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["Exclude Out Of Stock"]
                        }
                    },
                    "color": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["Black", "White", "Blue", "Gray", "Purple", "Pink", "Silver", "Gold", "Yellow"]
                        }
                    }
                },
                "required": ["brand", "os", "special_offers", "price", "series", "condition", "availability", "color"]
            }
        }
    ],
    [
        "5", "get_sort_by", "LOCAL_TABLET",
        {
            "name": "get_sort_by",
            "description": "Parameters to sort the products",
            "parameters" : {
                "type": "object",
                "properties": {
                    "sort_by": {
                        "type": "string",
                        "enum": ["Featured", "Best Sellers", "Newest", "Price Low to High", "Price High to Low"]
                    },
                },
                "required":["sort_by"],
            },
        }
    ]
]
###FIXME### Make sure it works
action_df = pd.DataFrame(actions, columns=['id', 'function_name', 'local_state', 'openai_func_call'])
###FIXME### create df of: |search|local_state|url|
page_nav = [
    [
        "Personal Phone", "LOCAL_PER_PHONE", "https://www.verizon.com/smartphones/"
    ],
    [
        "Tablets & Laptops", "LOCAL_TABLET", "https://www.verizon.com/tablets/"
    ]
]
###FIXME### Complete this and make sure it works
page_nav_df = pd.DataFrame(page_nav, columns=["search", "local_state", "url"]) 

def create_embedding(text):
    '''Create embeddings using ada-002'''

    response = openai.Embedding.create(
      model="text-embedding-ada-002",
      input=text
    )
    return response['data'][0]['embedding']

def generate_summary(q, context):
    '''Generate summary using query and context
    Serves as helper for summarize function
    Uses GPT-3.5-turbo with 16K context window'''

    query = f'''Given the provided #context#, summarize the answer to the user's #query# in under 200 words. Rely primarily on the #context# for the response.

###QUERY###
{q}

###CONTEXT###
{context}'''
    
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[{"role": "user", "content": query}], temperature=0.3)
    print(completion)###FIXME### Create log instead of printing
    return completion.choices[0].message.content

def summarize(query):
    '''Create a summary based on a user query'''

    #Get context from Vector DB
    pinecone.init(api_key="e7cb6b9a-88f4-46f9-a154-68a9f5feef72", environment="gcp-starter")
    vecdb = pinecone.Index("summarization")
    query_embedding = create_embedding(query) #vector embedding of query
    matches = vecdb.query(
        top_k=10,
        include_values=False,
        vector=query_embedding
    )
    ids = [int(x['id']) for x in matches['matches']] #ids of top k matches
    #context from topk using page raw text
    queried_dfs = summary_df[summary_df['id'].isin(ids)]
    context = ""
    for idx, r in queried_dfs.iterrows():
        context+=r['raw text']
    summary = generate_summary(query, context)
    links = [r['url'] for idx, r in queried_dfs.iterrows()]
    return (summary, links)

def intent_detection(query):
    '''Get intent from given query'''

    query = f'''Given the user query "#QUERY#", identify the intent and provide a response in the format: {{"intent": "INTENT", "action": "ACTION"}}. 
The possible #INTENT# values are:
- Information: Where the user is seeking an explanation, summary, or information.
- Action: Where the user intends to perform an action, like filtering, sorting, navigating etc.
- None: If there's no discernible intent.

If the #INTENT# is "Action", further describe the specific intent in #ACTION#. Some examples of navigation might be: "navigate to xyz.com", "show all phones", "scroll up" etc.

For instance, the query "Show me all iPhones in red color" would have a response as {{"intent": "Action", "action": "Show me phones"}}.

#QUERY#: "{query}"'''
    
    intent = openai.Completion.create(model="gpt-3.5-turbo-instruct", prompt=query, temperature=0.1)
    return intent.choices[0].text[2:]

def get_function_call(query_embedding, h_state):
    '''Return the Function Call ID based on embedding and state (Vector Search)'''

    pinecone.init(api_key="67c9dbd6-4fe6-4693-bcc4-fd1a9ff6357e", environment="gcp-starter")
    vecdb = pinecone.Index("action")
    matches = vecdb.query(
            top_k=1,
            include_values=False,
            vector=query_embedding,
            filter={
                "STATE": h_state
            }
        )
    return matches['matches'][0]["id"]

def get_action(query, query_intent_action, local_page):
    '''Return the action to be performed. Return NONE if no action'''

    query_updated = f'''For the following user #QUERY#, use function calling to see if a function should be used or not. If no function is used, return NONE

###QUERY###
{query}'''
    
    #Local Action
    if local_page in page_nav_df["url"].tolist():
        query_embedding = create_embedding(query)
        h_state = page_nav_df.loc[page_nav_df['url'] == local_page]['local_state'].values[0]
        function_call_id = get_function_call(query_embedding, h_state)

        response = openai.ChatCompletion.create(model="gpt-3.5-turbo",  temperature=0.1, functions=[action_df.loc[action_df['id'] == function_call_id]['openai_func_call'].values[0]], messages=[{"role": "user", "content": query_updated}])
        if "function_call" not in response.choices[0].message:
            return None
        return response
    #Global Action if 1. local action not hit 2. local action not available
    else:
        query_intent_action_embedding = create_embedding(query_intent_action)
        function_call_id = get_function_call(query_intent_action_embedding, "GLOBAL")
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", temperature=0.1, functions=[action_df.loc[action_df['id'] == function_call_id]['openai_func_call'].values[0]], messages=[{"role": "user", "content": query}])
        if "function_call" not in response.choices[0].message:
            return None
        #Check which global action was hit and act accordingly
        ###FIXME### Do something for scrolling
        if response.choices[0].message.function_call.name == "scroll":
            pass
        elif response.choices[0].message.function_call.name == "search":
            search_arg = json.loads(response.choices[0].message.function_call.arguments)['product']
            local_page = page_nav_df.loc[page_nav_df['search'] == search_arg]['url'].values[0]
            return get_action(query, query_intent_action, local_page)
        else:
            return None


def main_entry_function(query, local_page):
    '''Main usable function for singular query'''
    #Get intent
    query_intent = json.loads(intent_detection(query))
    #Conditional Statament
    if query_intent["intent"] == "Information":
        info = summarize(query)
        return info
    elif query_intent["intent"] == "Action":
        query_intent_action = query_intent['action']
        action = get_action(query, query_intent_action, local_page)
        return action
    else:
        pass

'''
VECTOR DB CREATION FUNCTIONS
'''

def create_summary_record(idx, ebd):
    '''Insert a record into the summary vector db'''
    pinecone.init(api_key="e7cb6b9a-88f4-46f9-a154-68a9f5feef72", environment="gcp-starter")
    vecdb = pinecone.Index("summarization")
    
    vecdb.upsert([(str(idx), ebd)])

def create_action_record(idx, h_state, ebd):
    '''Insert a record into the action vector db'''
    pinecone.init(api_key="67c9dbd6-4fe6-4693-bcc4-fd1a9ff6357e", environment="gcp-starter")
    vecdb = pinecone.Index("action")

    vecdb.upsert([(idx, ebd, {"STATE": h_state})])

def create_summary_vectordb():
    for idx, r in summary_df.iterrows():
        create_summary_record(r['id'], create_embedding(r['raw text']))

def create_action_vectordb():
    for idx, r in action_df.iterrows(): 
        create_action_record(r['id'], r['local_state'], create_embedding(r['openai_func_call']['description']))