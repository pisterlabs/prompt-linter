import openai
from termcolor import colored
import pandas as pd
import re
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
    # print(f"Selected {len(chosen_sections)} document sections:")
    # print("\n".join(chosen_sections_indexes))
    # print(type("".join(chosen_sections) ))
    # header = """Answer the question as truthfully as possible using the provided Given Below Information and previous conversation, and if the answer is not contained within in both, say "I don't know.For more details call on our CUSTOMER CARE NO.1800 1080"\n\nGiven Below Information:"""
    header = """Answer the question as truthfully as possible using the provided Given Below Information only,\n And if the answer is not contained within the text below, say "I don't have that information.For more details call on our CUSTOMER CARE NO.1800 1080 or visit our website https://www.icicibank.com/ and Don't justify your answers. Don't give information not mentioned in\nDon't make any change in urls\nif there any urls like website and image in Given Data must give it with response \n Given Below Information:\n"""

    return header + "".join(chosen_sections)

# conversation_history = []
def get_search_results(prompt):
    latest_question = prompt
    search_content = construct_prompt(latest_question,document_embeddings,df)
    # print("SEARCH    ",search_content)
        # get_redis_results(redis_client,latest_question,INDEX_NAME)['result'][0]
        
    return search_content
conversation = []
def ask_assistent(prompt):
    

    # if len(conversation) >=1:
            question_extract = openai.Completion.create(model="text-davinci-003",
                                                        prompt="""Generate a Customized Question based on Conversation
                                                        
For Example: 
question: List of ICICI credit card
conversation: []
generated_question: What is list of icici credit card available?

question: give me a detail of 8th credit card
conversation: [{'question': '\n\nWhat is list of icici credit card available?', 'response': 'ICICI Bank offers a variety of credit card options including:\n\n1. ICICI Bank Rubyx Credit Card\n2. ICICI Bank Sapphiro Credit Card\n3. ICICI Bank Coral Credit Card\n4. ICICI Bank Platinum Credit Card\n5. Manchester United Platinum Credit Card\n6. Manchester United Signature Credit Card\n7. Chennai Super Kings Credit Card\n8. MakeMyTrip ICICI Bank Platinum Credit Card\n9. MakeMyTrip ICICI Bank Signature Credit Card\n10. Unifare Mumbai Metro Cards\n11. Unifare Delhi Metro Cards\n12. Unifare Banglore Metro Cards\n13. Expressions Card\n14. HPCL Coral Visa Credit Card\n15. Emirates Credit Card\n16. Accelero Credit Card\n17. Amazonpay credit card \n\nTo know more about these credit cards, please visit https://www.icicibank.com/card/credit-cards/credit-card'}]
generated_question: Give me a details of MakeMy Trip ICICI bank Platinum Credit card

question : What is Home loan?
conversation:[{'question': '\n\nWhat are the different types of ICICI Credit Cards available?', 'response': 'ICICI Bank offers a variety of credit card options including:\n\n1. ICICI Bank Rubyx Credit Card\n2. ICICI Bank Sapphiro Credit Card\n3. ICICI Bank Coral Credit Card\n4. ICICI Bank Platinum Credit Card\n5. Manchester United Platinum Credit Card\n6. Manchester United Signature Credit Card\n7. Chennai Super Kings Credit Card\n8. MakeMyTrip ICICI Bank Platinum Credit Card\n9. MakeMyTrip ICICI Bank Signature Credit Card\n10. Unifare Mumbai Metro Cards\n11. Unifare Delhi Metro Cards\n12. Unifare Banglore Metro Cards\n13. Expressions Card\n14. HPCL Coral Visa Credit Card\n15. Emirates Credit Card\n16. Accelero Credit Card\n17. Amazonpay credit card \n\nTo know more about these credit cards, please visit https://www.icicibank.com/card/credit-cards/credit-card'}, {'question': '\n\nWhat are the features and benefits of the 8th credit card - MakeMyTrip ICICI Bank Platinum Credit Card?', 'response': 'The features and benefits of the MakeMyTrip ICICI Bank Platinum Credit Card are:\n\n1. Joining Fee: Rs.500+GST (one-time)\n2. Annual Fee (second year onwards): Nil\n3. Rs. 500 My Cash plus MakeMyTrip holiday voucher worth Rs. 3,000 as joining benefit.\n4. You get 2 MakeMyTrip vouchers worth Rs.1200 each on annual spends of Rs.1,00,000 and Rs.2,00,000\n5. You can enjoy 1 complimentary airport lounge access per quarter and 1 complimentary railway lounge access\n6. You can earn up to 5 MakeMyTrip Reward Points on every Rs.100 spent on the card, except fuel.\n7. The reward points earned on this card can be redeemed for bookings at MakeMyTrip website.\n8. A 1% fuel surcharge waiver can be availed on all fuel transactions at HPCL petrol pumps.\n9. Overdue interest rates are Monthly-3.50%, Annual-42.00%\n10. Visit this link https://www.icicibank.com/card/credit-cards/credit-card/makemytrip/platinum-credit-card to know more about MakeMyTrip ICICI Bank Platinum Credit Card.'}]
generated_question: What is Home Loan?

Note :As you see Home Loan topic is not releated previous conversation so asnwer as same as user asked

""" + """

question: {prompt}
Analyse this question and what user wants to know using keywords
conversation:{conversation}
Generate a Simple Question based on Conversation
generated_question:
""".format(prompt=prompt, conversation=conversation),max_tokens=50)
# f"Please generate a complete and clear question based on user's latest conversation: {conversation}.Extract it as a sentence starting the Question.Please ensure that the response is accurate and informative, and that it uses proper grammar and punctuation"
            # conversation.append(prompt)
            question = question_extract['choices'][0]['text']
            
            # print("QUE  ",question)
            # print("CON" ,  conversation)
            # print("Length" , len(conversation))
            return question
            
         
# Please generate a complete question based on the user's latest inquiry in
# the following conversation: [insert conversation here]. If the user has asked for a comparison between
# two specific ICICI Bank credit cards, please provide a clear and concise answer that highlights the k
# ey differences between those cards. If the user has asked for details about a specific card, please provide a comprehensive overview of the feature
# s, benefits, and eligibility criteria for that card. Please ensure that the response is accurate and informative, and that it uses proper grammar and punctuation.
       
       
def answer_query_with_context(
        query: str,
        df: pd.DataFrame,
        # document_embeddings: document_embeddings,
        # show_prompt: bool = False,
) -> str:
    
    # Construct prompt to send to OpenAI API
    # prompt = construct_prompt(
    #     query,
    #     document_embeddings,
    #     df,
    #     # conversation
    # )
    
    # Print prompt if show_prompt is True
    # if show_prompt:
    #     print(prompt)
     
    # try: 
        conversation_history = []
        # Send prompt to OpenAI API and get response
        while(True):
            # user_input = input("")  
            query1 = ask_assistent(query)
            print("QUestion" , query1)
            system = get_search_results(query1)
            message = {"role": "system", "content": system}  
            conversation_history.append(message)
            user = {"role": "user", "content": query1}
            conversation_history.append(user)
            # conversation.append({"content": query1})
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages = conversation_history
            )

            conversation_history.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
            conversation.append({"question":query1,"response": response['choices'][0]['message']['content']})
            # print("Conve",conversation_history)
            # print("\n" + response['choices'][0]['message']['content'] + "\n")
            
            text = response['choices'][0]['message']['content']
        
            regex = r"(?P<url>https?://[^\s]+)"
            #     regex = r'https?://(?:www\.)?icicibank\.com/\S+(?:\?\S*)?(?:#\S*)?'
            #     regex = r"https?://[^\s<>]+(?:\w/)?(?:[^\s()]*)"
            match = re.search(regex, text)
            # print(match)
            # if match:
            #     url = match.group("url")
            #     text = text.replace(url, "")
            #     # link = generate_response(url)
            #     link = url
            #     #         print(link)
            #     return f"{text}, {link}"

            # Return response text
            return text
    # except openai.error.APIError as e:
    #      #Handle API error here, e.g. retry or log
    #     return f"OpenAI API returned an API Error: {e}"
        
    # except openai.error.APIConnectionError as e:
    #     #Handle connection error here
    #     return f"Failed to connect to OpenAI API: {e}"
        
    # except openai.error.RateLimitError as e:
    #     #Handle rate limit error (we recommend using exponential backoff)
    #     return f"OpenAI API request exceeded rate limit: {e}"
        
    
def inputdata(inpval: str) :
    response = answer_query_with_context(inpval, df)
    if isinstance(response, tuple):
        text, url = response
        return text, url
    else:
        return response
    

       
        

