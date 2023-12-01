
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
import json
import numpy as np
from langchain.schema import AIMessage,HumanMessage,SystemMessage

chat = ChatOpenAI(temperature=0,model='gpt-3.5-turbo')

messages = [
    SystemMessage(content="You are a helpful assistant"),
]

def get_context(inputPrompt,embeddings):
    # openai.api_key = apiKey
    search_term_vector = embeddings
    
    with open("knowledge.json",encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
        for item in data:
            item['embeddings'] = np.array(item['embeddings'])

        for item in data:
            item['similarities'] = cosine_similarity(item['embeddings'], search_term_vector)

        sorted_data = sorted(data, key=lambda x: x['similarities'], reverse=True)
        context = ''
        referencs = []
        for i in sorted_data[:3]:
            context += i['chunk_text'] + '\n'
            
    return context



while True:
    print("---- Make sure to subscribe d dot py:  https://www.youtube.com/@ddotpy")
    user_input = input("you: ")

    embd = get_embedding( user_input,engine='text-embedding-ada-002')

    context = get_context(user_input,embd)
    


    messages.append(HumanMessage(content="context: \n" +context+"\n\nplease answer the following question using the above given context\n\n"+user_input))
    ai_response = chat(messages=messages).content
    print("ai: ",ai_response)
    messages.append(AIMessage(content=ai_response))

