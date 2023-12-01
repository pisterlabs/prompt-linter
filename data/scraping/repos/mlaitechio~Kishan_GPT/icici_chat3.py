import openai
from termcolor import colored
import pandas as pd
import tiktoken
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY") 
openai.api_key = API_KEY
# import streamlit as st
# from .icici_chat import construct_prompt
# from database import get_redis_connection,get_redis_results

# from config import CHAT_MODEL,COMPLETIONS_MODEL, INDEX_NAME

# redis_client = get_redis_connection()
EMBEDDING_MODEL = "text-embedding-ada-002"
# A basic class to create a message as a dict for chat

df = pd.read_csv('icici_with_token_new.csv')

# df["token"] = None
# for idx, r in df.iterrows():
# #     print(len(r.content))
# #     df["token"] = df[len(r.content)]
#     df.loc[idx,'token'] = len(r.content)

# "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" -- MLAI
# "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" -- MLAI
# "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" -- LP
def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
 
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

# import time
# def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
#     embeddings = {}
#     for idx, r in df.iterrows():
#         print(idx)
# #         print(r)
#         embeddings[idx] = get_embedding(r.title)
#         time.sleep(5)  # Add a delay of 10 second between requests
#     return embeddings


def load_embeddings(fname: "str") -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.

    fname is the path to a CSV with exactly these named columns:
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """

    df = pd.read_csv(fname, header=0)

    max_dim = max([int(c) for c in df.columns if c != "title"])
    return {
        (r.title): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()

    }

document_embeddings = load_embeddings("icici_embed.csv")
def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.

    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))


def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[
    (float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)

    document_similarities = sorted([
        (cosine_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities

# a = order_document_sections_by_query_similarity("Give me a list of icici credit crads  interest rate", document_embeddings)[:5]

MAX_SECTION_LEN = 7552
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

print(f"Context separator contains {separator_len} tokens")


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)[:3]
    #     print(most_relevant_document_sections)
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        #         print(section_index)
        document_section = df.loc[section_index]
        #         print(document_section)
        chosen_sections_len += document_section.token + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))

    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
    # print(type("".join(chosen_sections) ))
    # header = """Answer the question as truthfully as possible using the provided Given Below Information and previous conversation, and if the answer is not contained within in both, say "I don't know.For more details call on our CUSTOMER CARE NO.1800 1080"\n\nGiven Below Information:"""
    header = """Answer the question as truthfully as possible using the provided Given Below Information only, and if the answer is not contained within the text below, say "I don't know.For more details call on our CUSTOMER CARE NO.1800 1080 or visit our website https://www.icicibank.com/"\n\nGiven Below Information:\n"""

    return header + "".join(chosen_sections) 

class Message:
    
    
    def __init__(self,role,content):
        
        self.role = role
        self.content = content
        
    def message(self):
        
        return {"role": self.role,"content": self.content}
# Updated system prompt requiring Question and Year to be extracted from the user

# New Assistant class to add a vector database call to its responses
class RetrievalAssistant:
    
    def __init__(self):
        self.conversation_history = []  

    def _get_assistant_response(self, prompt):
    #     que = construct_prompt(
    #     prompt,
    #     document_embeddings,
    #     df,
    #     # conversation
    # )
        print(prompt)
        
        try:
            completion = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages=prompt,
              temperature=0.1
            )
            
            response_message = Message(completion['choices'][0]['message']['role'],completion['choices'][0]['message']['content'])
            return response_message.message()
            
        except Exception as e:
            # print(prompt)
            return f'Request failed with exception {e}'
    
    # The function to retrieve Redis search results
    def _get_search_results(self,prompt):
        latest_question = prompt
        search_content = construct_prompt(latest_question,document_embeddings,df)
        # print("SEARCH    ",search_content)
        # get_redis_results(redis_client,latest_question,INDEX_NAME)['result'][0]
        
        return search_content
        

    def ask_assistant(self, next_user_prompt):
        [self.conversation_history.append(x) for x in next_user_prompt]
        assistant_response = self._get_assistant_response(self.conversation_history)
        # print(assistant_response)
        
        # Answer normally unless the trigger sequence is used "searching_for_answers"
        if len(self.conversation_history) >=1:
            question_extract = openai.Completion.create(model="text-davinci-003",prompt=f"create a question the user's latest question from this conversation: {self.conversation_history}.Extract it as a sentence stating the Question \n Example like if user's first question is list of icici card and in response there are all list of cards after that if user ask details of 2nd option so give question like details of name of 2nd card in list " )
            print("HEllo   ",question_extract['choices'][0]['text'])
            search_result = self._get_search_results(question_extract['choices'][0]['text'])
            # print("SEARCH    ",search_result)
            # We insert an extra system prompt here to give fresh context to the Chatbot on how to use the Redis results
            # In this instance we add it to the conversation history, but in production it may be better to hide
            # self.conversation_history.insert(-1,{"role": 'system',"content": f"Answer the {question_extract['choices'][0]['text']} question using this content: {search_result}. If you cannot answer the question, say 'Sorry, I don't know the answer to this one'"})
            self.conversation_history.insert(-1,{"role": 'system',"content": f"Don't take user's question {search_result}" +"\n\n Q: " +"refer to this"+ f"{question_extract['choices'][0]['text']}" + "\n A:"})

            assistant_response = self._get_assistant_response(self.conversation_history)
            
            self.conversation_history.append(assistant_response)
            return assistant_response
        else:
            self.conversation_history.append(assistant_response)
            return assistant_response
            
        
    def pretty_print_conversation_history(self, colorize_assistant_replies=True):
        for entry in self.conversation_history:
            if entry['role'] == 'system':
                pass
            else:
                prefix = entry['role']
                content = entry['content']
                output = colored(prefix +':\n' + content, 'green') if colorize_assistant_replies and entry['role'] == 'assistant' else prefix +':\n' + content
                #prefix = entry['role']
                # print(output)
                
conversation = RetrievalAssistant()

# Create a list to hold our messages and insert both a system message to guide behaviour and our first user question
messages = []
# system_message = Message('system','You are a helpful business assistant who has innovative ideas')
user_message = Message('user','list of icici credit card')
# messages.append(system_message.message())
messages.append(user_message.message())
# # messages
# print(messages)

# Get back a response from the Chatbot to our question
response_message = conversation.ask_assistant(messages)
# print(response_message['content'])

next_question = 'Tell me more about 12 number card from list'

# # Initiate a fresh messages list and insert our next question
# messages = []
user_message = Message('user',next_question)
messages.append(user_message.message())
response_message = conversation.ask_assistant(messages)
print(response_message['content'])

