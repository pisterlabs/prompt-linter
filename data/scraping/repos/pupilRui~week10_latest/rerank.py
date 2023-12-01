#!/usr/bin/env python
# coding: utf-8

# # ReRank

# ## Setup
# 
# Load needed API keys and relevant Python libaries.

# In[2]:


# !pip install cohere 
# !pip install weaviate-client


# In[71]:


import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


# In[72]:


import cohere
co = cohere.Client(os.environ['COHERE_API_KEY'])


# In[73]:


import weaviate
auth_config = weaviate.auth.AuthApiKey(
    api_key=os.environ['WEAVIATE_API_KEY'])


# In[74]:


client = weaviate.Client(
    url=os.environ['WEAVIATE_API_URL'],
    auth_client_secret=auth_config,
    additional_headers={
        "X-Cohere-Api-Key": os.environ['COHERE_API_KEY'],
    }
)


# ## Dense Retrieval

# In[77]:


from utils import dense_retrieval


# In[78]:


query = "What is the capital of Canada?"


# In[126]:


dense_retrieval_results = dense_retrieval(query, client)


# In[127]:


from utils import print_result


# In[128]:


print_result(dense_retrieval_results)


# ## Improving Keyword Search with ReRank

# In[84]:


from utils import keyword_search


# In[85]:


query_1 = "What is the capital of Canada?"


# In[112]:


query_1 = "What is the capital of Canada?"
results = keyword_search(query_1,
                         client,
                         properties=["text", "title", "url", "views", "lang", "_additional {distance}"],
                         num_results=3
                        )

for i, result in enumerate(results):
    print(f"i:{i}")
    print(result.get('title'))
    print(result.get('text'))


# In[ ]:


query_1 = "What is the capital of Canada?"
results = keyword_search(query_1,
                         client,
                         properties=["text", "title", "url", "views", "lang", "_additional {distance}"],
                         num_results=500
                        )

for i, result in enumerate(results):
    print(f"i:{i}")
    print(result.get('title'))
    #print(result.get('text'))


# In[113]:


def rerank_responses(query, responses, num_responses=10):
    reranked_responses = co.rerank(
        model = 'rerank-english-v2.0',
        query = query,
        documents = responses,
        top_n = num_responses,
        )
    return reranked_responses


# In[114]:


texts = [result.get('text') for result in results]
reranked_text = rerank_responses(query_1, texts)


# In[115]:


for i, rerank_result in enumerate(reranked_text):
    print(f"i:{i}")
    print(f"{rerank_result}")
    print()


# ## Improving Dense Retrieval with ReRank

# In[116]:


from utils import dense_retrieval


# In[117]:


query_2 = "Who is the tallest person in history?"


# In[130]:


results = dense_retrieval(query_2,client)


# In[132]:


for i, result in enumerate(results):
    print(f"i:{i}")
    print(result.get('title'))
    print(result.get('text'))
    print()


# In[121]:


texts = [result.get('text') for result in results]
reranked_text = rerank_responses(query_2, texts)


# In[122]:


for i, rerank_result in enumerate(reranked_text):
    print(f"i:{i}")
    print(f"{rerank_result}")
    print()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



