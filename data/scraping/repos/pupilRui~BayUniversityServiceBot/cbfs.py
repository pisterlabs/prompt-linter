from langchain.embeddings.openai import OpenAIEmbeddings
import openai
import os
import sys
import requests
import uuid
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['API_KEY']
goog_api_key    = os.environ['GOOG_API_KEY']

sentence1 = "What are the application requirements for the MSCS program at SFBU?"
sentence2 = "Can you outline the core curriculum for the SFBU MSCS program?"
sentence3 = "What are the tuition and fees for the MSCS program at SFBU?"
sentence4 = "Are there any prerequisite courses or experience required for the MBA program at SFBU?"
sentence5 = "What specializations are available within the SFBU MBA program?"
sentence6 = "How do I contact the admissions office for the MBA program at SFBU?"
sentence7 = "What are the graduation requirements for the MSCS program at SFBU?"
sentence8 = "Can you provide information on the faculty for the MBA program at SFBU?"
sentence9 = "Are there part-time or online options for the MSCS or MBA programs at SFBU?"
sentence10 = "What career services are available for MSCS and MBA students at SFBU?"

embedding = OpenAIEmbeddings(openai_api_key=openai.api_key)
embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)
embedding4 = embedding.embed_query(sentence4)
embedding5 = embedding.embed_query(sentence5)
embedding6 = embedding.embed_query(sentence6)
embedding7 = embedding.embed_query(sentence7)
embedding8 = embedding.embed_query(sentence8)
embedding9 = embedding.embed_query(sentence9)
embedding10 = embedding.embed_query(sentence10)


# In[ ]:


import numpy as np


# In[ ]:


# numpy.dot(vector_a, vector_b, out = None) 
# returns the dot product of vectors a and b.
np.dot(embedding1, embedding2)


# In[ ]:


np.dot(embedding1, embedding3)


# In[ ]:


np.dot(embedding2, embedding3)



#############################################################
# 4. Vectorstores
#############################################################


# In[ ]:


# ! pip install chromadb


# In[ ]:


from langchain.vectorstores import Chroma


# In[ ]:


persist_directory = 'docs/chroma/'

# In[ ]:


vectordb = Chroma(
    embedding_function=embedding,
    persist_directory=persist_directory
)


# In[ ]:


# print(vectordb._collection.count())

from langchain.memory import ConversationBufferMemory



llm_name = "gpt-3.5-turbo"

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name=llm_name, temperature=0, openai_api_key=openai.api_key)
llm.predict("Hello world!")

from langchain.chains import ConversationalRetrievalChain

retriever=vectordb.as_retriever()

def load_db():
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        # Set return messages equal true
        # - Return the chat history as a  list of messages 
        #   as opposed to a single string. 
        # - This is  the simplest type of memory. 
        #   + For a more in-depth look at memory, go back to  
        #     the first class that I taught with Andrew.  
        return_messages=True
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
    )
    return qa

import panel as pn
import param

#############################################################
# Step 7.1.2: cbfs class
#############################################################
class cbfs(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query  = param.String("")
    db_response = param.List([])
    
    #########################################################
    # Step 7.1.2.1: init function
    #########################################################
    def __init__(self,  **params):
        super(cbfs, self).__init__( **params)
        self.panels = []
        self.qa = load_db()
    
    #########################################################
    # Step 7.1.2.2: call_load_db function
    #########################################################
    def call_load_db(self, count):
        self.clr_history()
        self.qa = load_db()

    #########################################################
    # Step 7.1.2.3: convchain(self, query) function
    #########################################################
    def convchain(self, query):
        if not query:
            return pn.WidgetBox(pn.Row('User:', 
               pn.pane.Markdown("", width=600)), scroll=True)
        result = self.qa({"question": query, 
                          "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        return result["answer"]


    #########################################################
    # Step 7.1.2.7: clr_history function
    #########################################################
    def clr_history(self):
        self.chat_history = []
        self.qa = load_db()
        return