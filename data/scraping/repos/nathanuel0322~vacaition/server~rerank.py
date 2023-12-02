import cohere, os
from environment import ENVIRONMENT
from firebaseInterface import getAllUserInfo
import json

# Get your cohere API key on: www.cohere.com
co = cohere.Client(os.environ["COHERE_KEY"])

# Example query and passages
# query = "I like to do hiking and going to the beach"

# use rerank to get the top buddies from their preferences and comments
# There are hard restrictions and then their are soft restrictions where we rank the top 3
def get_buddies(u_id, user_info):
    # Need to use the user_info to get the preferences and comments
    other_user_info = getAllUserInfo(u_id)
    print(other_user_info)
    documents = []
    for user in other_user_info:
        documents.append(json.dumps(user))
    results = co.rerank(query=str(user_info), 
                        documents=documents, top_n=3, model="rerank-multilingual-v2.0")

    return results