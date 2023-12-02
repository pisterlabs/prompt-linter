#!/usr/bin/env python
# coding: utf-8

################################################
# ReRank
#
# Step 1: Import two libries: cohere and weaviate
# Step 2: Apply Dense Retrieval to a query
# Step 3: Improving Keyword Search with ReRank
# Step 4: Improving Dense Retrieval with ReRank
################################################

# ## Setup
#
# Load needed API keys and relevant Python
# libaries.

# In[2]:


# !pip install cohere
# !pip install weaviate-client


# In[71]:


from utils import keyword_search
from utils import print_result
from utils import dense_retrieval
import weaviate
import cohere
import os

# read local .env file
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


# In[72]:

################################################
# 1. Import two libries
################################################


################################################
# 1.1 Import cohere
################################################
co = cohere.Client(os.environ['COHERE_API_KEY'])


# In[73]:


################################################
# 1.2 Import weaviate
################################################
auth_config = weaviate.auth.AuthApiKey(
    api_key=os.environ['WEAVIATE_API_KEY'])


# In[74]:


client = weaviate.Client(
    url=os.environ['WEAVIATE_API_URL'],
    auth_client_secret=auth_config,
    additional_headers={
        "X-Cohere-Api-Key":
        os.environ['COHERE_API_KEY'],
    }
)


# ################################################
# # 2. Dense Retrieval
# ################################################

# # In[77]:


# # In[78]:


query = "What is the capital of Canada?"


# # In[126]:


# ################################################
# # 2.1 Apply Dense Retrieval to a query
# ################################################
dense_retrieval_results = dense_retrieval(query,
                                          client)


# # In[127]:


# # In[128]:


# ################################################
# # 2.2 Print the result of the Dense Retrieval to
# #     a query
# ################################################
# print_result(dense_retrieval_results)


# ################################################
# # 3. Improving Keyword Search with ReRank
# ################################################

# In[84]:


# In[85]:


query_1 = "What is the capital of Canada?"


# In[112]:


################################################
# 3.1 Keyword Search with 3 results
################################################
query_1 = "What is the capital of Canada?"
results = keyword_search(query_1,
                         client,
                         properties=["text", "title", "url", "views",
                                     "lang",
                                     "_additional {distance}"],
                         num_results=3
                         )

# for i, result in enumerate(results):
#     print(f"i:{i}")
#     print(result.get('title'))
#     print(result.get('text'))


# In[]:

################################################
# 3.2 Keyword Search with 500 results
################################################
query_1 = "What is the capital of Canada?"
results = keyword_search(query_1,
                         client,
                         properties=["text", "title", "url", "views",
                                     "lang",
                                     "_additional {distance}"],
                         num_results=500
                         )

# for i, result in enumerate(results):
#     print(f"i:{i}")
#     print(result.get('title'))
#     # print(result.get('text'))


# # In[113]:


# ################################################
# # 3.3 ReRank of the Keyword Search results
# ################################################
def rerank_responses(query, responses,
                     num_responses=10):
    reranked_responses = co.rerank(
        model='rerank-english-v2.0',
        query=query,
        documents=responses,
        top_n=num_responses,
    )
    return reranked_responses


# # In[114]:


texts = [result.get('text') for result in
         results]
reranked_text = rerank_responses(query_1,
                                 texts)


# In[115]:


# for i, rerank_result in enumerate(reranked_text):
#     print(f"i:{i}")
#     print(f"{rerank_result}")
#     print()

# ################################################
# # 4. Improving Dense Retrieval with ReRank
# ################################################

# # In[116]:


# # In[117]:


query_2 = "Who is the tallest person in history?"


# In[130]:


################################################
# 4.1 Dense Retrieval of a new query
################################################
results = dense_retrieval(query_2, client)


# In[132]:


# for i, result in enumerate(results):
#     print(f"i:{i}")
#     print(result.get('title'))
#     print(result.get('text'))
#     print()


# # In[121]:


################################################
# 4.2 ReRank the Dense Retrieval of a
#     new query
################################################
texts = [result.get('text') for result
         in results]
reranked_text = rerank_responses(query_2,
                                 texts)


# In[122]:


for i, rerank_result in enumerate(
        reranked_text):
    print(f"i:{i}")
    print(f"{rerank_result}")
    print()
