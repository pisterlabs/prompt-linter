#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install langchain


# In[2]:


# import libraries
import os
from langchain.llms import Cohere


# In[ ]:


pip install cohere


# In[5]:


llm = Cohere(temperature = 0, cohere_api_key="4aJ9yWbIrOzI2W5LZeLeIdin2AYMpkq18PffLuvi")


# In[7]:


api_key="765d5794f78a2cfbd3ee70f8ce4d43d36a111400e750d27eaf5721f2703624e9"


# In[ ]:


get_ipython().system('pip install google-search-results')


# In[8]:


from langchain.agents import load_tools
from langchain.agents import initialize_agent


# In[10]:


tools = load_tools(["serpapi", "llm-math"], llm=llm, serpapi_api_key=api_key)


# In[11]:


agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)


# In[ ]:


agent.run("""What is the height of Obama?
And how many cans of coke can you stack to reach that height?""")

