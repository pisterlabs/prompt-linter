import re
import openai
# get configuration
# use for get number of tokens
from transformers import GPT2Tokenizer
# essentials
import numpy as np
import math
# get secrets
import streamlit as st


# instance tokenizer & api
openai.api_key = st.secrets["GPT3"]["OPENAI_API_KEY"] #os.getenv("OPENAI_API_KEY")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
## Reducing text
def reduce_text_to_nearest_period(text, MAX_LEN):
    """ Reduce text to nearest period always the text is less than MAX_LEN
    """
    dots_idx = np.array( find_words_endswith_dot(text) )
    # select dot position less than MAX_TOKEN
    best_dot_idx = dots_idx[ (dots_idx - MAX_LEN)<0 ]
    if len(best_dot_idx) == 0:
        reduce_text = text[:MAX_LEN]
    else:
        reduce_text = text[:best_dot_idx[-1]+1]
    return reduce_text


# create function to search postions of "." in list_words
def find_words_endswith_dot(list_words):
    pos = []
    for i in range(len(list_words)):
        if list_words[i].endswith("."):
            pos.append(i)
    return pos
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
## Reduce tokens for using gpt3 api
def reduce_tokens_for_gpt3(input_text_gpt3):
    """reduce tokens for using gpt3 api
    """
    # if text have periods --->  reduce using periods
    best_id_dot = 0
    dots_idx = find_words_endswith_dot(input_text_gpt3)
    for id in dots_idx:
        number_tokens_text = get_number_of_tokens(input_text_gpt3[:id+1])
        if number_tokens_text < st.secrets["GPT3"]["MAX_TOKENS"]:
            best_id_dot = id

    if best_id_dot != 0:
        reduce_text = input_text_gpt3[:best_id_dot+1]
        return reduce_text

    else: # if text dont have period ---> force reduce
        reduce_text = ""
        for word in input_text_gpt3.split(" "):
            if get_number_of_tokens(reduce_text) < st.secrets["GPT3"]["MAX_TOKENS"] - 1:
                reduce_text  = reduce_text + " " +  word
            else:
                break
        return reduce_text

def get_number_of_tokens(text):
    return len(tokenizer(text)['input_ids'])
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Main function
def get_ingerence_GPT3(input_text, MODE, ENGINE, LANGUAGE="English"):
    # number of tokens would be less than st.secrets["GPT3"]["MAX_TOKENS"]
    reduce_text = reduce_tokens_for_gpt3(input_text)

    # define prompt
    # define prompt
    if LANGUAGE == "English":
        command = st.secrets["FUNCTIONS"][MODE]["PROMPT"]
    elif LANGUAGE == "Spanish":
        command = st.secrets["FUNCTIONS"][MODE]["PROMPT_ES"]
    prompt = command[0] + reduce_text + command[1]

    response_raw = openai.Completion.create(
        engine= st.secrets["GPT3"]["ENGINE_MODEL"][ENGINE]["MODEL"],
        prompt=prompt,
        temperature=0,
        max_tokens=st.secrets["FUNCTIONS"][MODE]["MAX_TOKENS"],
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        #stop=["\n"],
        #n=3,
        #best_of=1,
    )
    # get raw text
    response = response_raw["choices"][0]["text"]
    # clean blank space and brea lines
    response = response.replace("\n","").strip()
    # clean tags 
    if MODE == "TAGS":
        response = response[:-1] if response.endswith(",") else response

    # get price for prediction
    tokens_1000 = math.ceil( get_number_of_tokens( reduce_text + " " + response_raw["choices"][0]["text"] )/1000 )
    price = tokens_1000 * st.secrets["GPT3"]["ENGINE_MODEL"][ENGINE]["PRICE"] # USD

    return response, price

if __name__ == "__main__":
    # inputs 
    input_text = "Agent: For calling customer service. My name is Vanessa, how may help you. Client: I was calling to order place and white. Agent: We happy to send out a replacement card out for you. Agent: Your 16, digit card number. Client: I dont know the Cardinals. Agent: Thank you verify your first and last name please. Client: Patricia Covington. Agent: How you spell your last name? Client: C O V I N G T O and. Agent: And you said your first name. Client: Letricia. Agent: L a T. Client: R I C I. Agent: Z. Agent: It's not pulling up anything C O N C I N G T O. Client: Know C O V as in Victor, Agent: S when? Client: I N G T O N E. Agent: I put the extra letter and I was wondering what I was doing wrong key verify your data birth for the reserve anson. Client: Uh huh made since 1995. Agent: Thing with this card last owner damage. Client: It was last. Agent: Thinking to verify your address we like a new cards remote out to. Client: 1918 Arlington avenue saint, Louis Missouri 63112 apartment a. Agent: You. Okay. Thank you Mrs. Could send him before? I cant see your car. I need to inform you that this call will be personally cancel excuse me. It will take three to five business days for your new card to arrive in the mail would you like him for counselors car now. Client: Yes maam. Agent: Thank you your card is now been council your my name is Alison team will be transferred to your new card you have 121 instead of benefits available and a dollar and 0.38 and cash benefits. Client: Okay. Thank you. I have you. Agent: Or anything else? I can assist you with today. Client: Know you have a good day. Agent: I was coming soon. Thank you for calling customer service and have a nice day. Client: Thank you, bye, bye. Thank you bye."
    LIST_MODE = ["TITLE" ,"SUMMARY", "TAGS", "SENTIMENT"]
    # Select Engine ---> Ada, Babbage, Curie, Davinci
    ENGINE = "Davinci"


    # get result in json format
    res = dict()
    res["ENGINE"] = ENGINE
    total_price = 0
    for MODE in LIST_MODE:
        response, price = get_ingerence_GPT3(input_text, MODE, ENGINE)
        res[MODE] = response
        total_price += price
    res["PRICE"] = total_price


    # show result
    from pprint import pprint
    print(f"Input_text:\n{input_text}")
    print("--------------------------------------\n")
    pprint(res)