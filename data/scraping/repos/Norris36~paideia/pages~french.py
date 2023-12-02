## Write a streamlit application that can translate english to french, or help me optimising my french

import streamlit as st
import pandas as pd
import openai
import numpy as np
import time
import random
import regex as re
from datetime import datetime
import os

openai.api_type         = os.getenv("OPENAI_TYPE")
openai.api_base         = os.getenv("OPENAI_BASE")
openai.api_version      = os.getenv("OPENAI_VERSION")
openai.api_key          = os.getenv("OPENAI_KEY")


def vector_similarity(vec1,vec2):
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(vec1), np.array(vec2))

def calculate_similarity(vector, prompt_embedding):
    #print(type(vector),type(prompt_embedding))
    similarity = vector_similarity(vector, prompt_embedding)
    return similarity

def get_embedding(text):
    tries = 0
    if isinstance(text, str):
        pass
    else:
        return None
    try:
        result = openai.Embedding.create(
            engine='text-embedding-ada-002',
            model='text-embedding-ada-002',
            input=text
        )
        return result["data"][0]["embedding"]
    except Exception as e:
        #print(f"Error: {e}")
        time.sleep(7)
        if tries < 3:
            get_embedding(text)
            tries += 1
            #print(tries)
        else:
            return None

def cleaner(text):
    if isinstance(text, str):
        list = []

        text = text.replace('[', '')
        text = text.replace(']', '')

        for j in text.split(','):
            j = j.strip()
            list.append(float(j))
        return list
    else:
        return text

def my_reader(path):
    embeddings = []
    try:
        df = pd.read_csv(path)
    except:
        df = pd.read_csv(path, lineterminator='\n')
    
    for i in range(len(df)):
        x = df.at[i, 'embedding']
        y = cleaner(x)
        embeddings.append(y)
        
    df['embedding'] = embeddings
    return df

def embed_prompt_lookup(sentence: str, df: pd.DataFrame, goal: str = '') -> str:
    #key = 'article'
    prompt_embedding = get_embedding(sentence)



    df['prompt_similarity'] = np.nan  

    # if len(key) > 3:
    #     definition              = df[df.Name_ == sample_name].text.values[0]
    #     definition_embedding    = df.loc[df.Name_ == sample_name, 'embedding'].values[0]
    #     prompt_similarity       = vector_similarity(definition_embedding,
    #                                                 prompt_embedding)
    #     most_similar_summary    = definition    
    # else:
    #     # Initialize a new column with NaN values

    for index, row in df.iterrows():
        if pd.notna(row['embedding']).all():
            df.at[index, 'prompt_similarity'] = vector_similarity(list(row['embedding']), prompt_embedding)

        # get most similar summary
    phrases = df.nlargest(3, 'prompt_similarity')['french'].to_list()
    #prompt_similarity = df.nlargest(1, 'prompt_similarity')['prompt_similarity'].values[0]

    # Rest of your code...

    prompt = f"""
            You will act as my french teacher, aiding me from A1 to A2 level.
            I will provide you with my attempt at {st.session_state.objective}, and you will correct me, based on the data i give, as i want the sentences to be very simple.
            But you can use other data, if you think it will help to preserve the meaning im striving for.
            You will receive, my attempt at writing french, and the sentences that inspired me from the phrasebook.
            Ensure that your feedback is based on the phrasebook, and the english version of the correct answer.
            Your reply must be in english, and follow the below format, as im not sufficent enough in french.
            The goal of the below sentence is to {goal}.
            
            The data:
            My attempt at writing french:
            {sentence}
            
            The sentences that inspired me:             
            {phrases[0]}
            
            The format should be:
            ---------------------
            English: [the sentence in english]
            My answer: [my attempt, untouched, at writing french]
            The correct: [the correct answer, in french]
            
            Feedback: [your feedback]
            ---------------------
            """
    
    if goal == '':
        prompt.replace('The goal of the below sentence is to {goal}.', '')
        
    # Use OpenAI to get the most similar answer
    response = openai.Completion.create(
        engine = 'text-davinci-003', # Use the text-davinci-003 engine
        prompt=prompt, # Use the prompt created above
        temperature=0, # Set the temperature to 0
        max_tokens=500, # Set the maximum tokens to 500
        model="text-davinci-003" # Use the text-davinci-003 model
        )
    response = response["choices"][0]["text"].strip(" \n")
    return response, phrases

def get_data_path():
   # The `return os.path.join(os.path.dirname(__file__), "data.csv")` line of code is returning the
   # full path of the `data.csv` file by joining the directory path of the current file with the
   # filename `data.csv`.
   return os.path.join(os.path.dirname(os.path.dirname(__file__)), "french_paleur.csv")


if 'output' not in st.session_state:
    st.session_state.output = ''

if 'input' not in st.session_state:
    st.session_state.input = ''

if 'path' not in st.session_state:
    st.session_state.path = r'C:\Users\jbay\OneDrive - GN Store Nord\Workspace\french\french_paleur.csv'

if 'search' not in st.session_state:
    st.session_state.search = 1

if 'df' not in st.session_state:
    df = my_reader(st.session_state.path)
    st.session_state.df = df
    st.session_state.mod = st.session_state.df[st.session_state.df.last_run < st.session_state.df.last_run.min() + 1].index.values[0]       
    
if 'next' not in st.session_state:
    st.session_state.next = False

if 'current_sentence' not in st.session_state:
    st.session_state.current_sentence = ""

# Set the title of the page
st.title("French Translator")

st.selectbox("What do you want to do?", ("Practicing Translating French to English",
                                        "Translate from English to French",
                                         "Optimise French",
                                         "Translate from French to English",
                                         "Practicing Translating English to French"), key = 'objective')

def update_sentence(language = 'french', force = False):
    if language == 'french':
        fact_lang = 'english'
    else:
        fact_lang = 'french'
        
    if st.session_state.search > 70 or st.session_state.next or force:           
        st.session_state.mod = st.session_state.df[st.session_state.df.last_run < st.session_state.df.last_run.min() + 1].index.values[0]       
        sentence  = st.session_state.df.at[st.session_state.mod, language]
        st.session_state.current_sentence = sentence
        st.session_state.fact = st.session_state.df[st.session_state.df[language] == sentence][fact_lang].values[0]
        st.session_state.next = False

if st.session_state.objective == "Practicing Translating French to English":
    update_sentence(force = True)
elif st.session_state.objective == "Practicing Translating English to French":
    update_sentence(language='english', force = True)

st.write(st.session_state.mod, st.session_state.current_sentence )
if st.button("Next"):
    st.session_state.next = True
    update_sentence(w)
    #st.write(st.session_state.current_sentence)
st.text_area("Input", key = 'input')

st.text_input("goal", key = 'goal')

if st.button("Translate"):
    if st.session_state.objective != "Practicing Translating French to English" and st.session_state.objective != "Practicing Translating English to French":
        response, phrases = embed_prompt_lookup(st.session_state.input, st.session_state.df, st.session_state.goal)
        st.session_state.output = response
    else:
        prompt = f"""
                    You will act as my french teacher, helping me for an exam, andaiding me from A1 to A2 level.
                    I will provide you with my attempt at {st.session_state.objective}, and you will correct me, based on the data i give, as i want the sentences to be very simple.
                    You will receive, my attempt, and the sentences that i tried to translate.
                    Ensure that your feedback is based on the phrasebook, and the english version of the correct answer.
                    You must provide a 0 / 100 score of my translation, Your score must be precise as it will impact my exam.
                    100 meaning perfect, 80 meaning really good, 60 meaning good, 40 meaning ok, 20 meaning bad, 0 meaning really bad.
                    Your reply must be in english, and follow the below format, as im not sufficent enough in french.
                    
                    
                    
                    The data:
                    My attempt at translating:
                    {st.session_state.input}
                    
                    The french sentence i tried to translate             
                    {st.session_state.current_sentence}
                    
                    The correct translation:
                    {st.session_state.fact}
                    
                    The format should be:
                    ---------------------
                    [your score]/100   
                    
                                     
                    My answer: [my attempt, untouched]
                    The correct translation: [the correct answer]
                    
                    Feedback: [your feedback]
                    ---------------------
                    A:
                    """
        st.session_state.search = 1        
        # Use OpenAI to get the most similar answer
        response = openai.Completion.create(
            engine = 'text-davinci-003', # Use the text-davinci-003 engine
            prompt=prompt, # Use the prompt created above
            temperature=0, # Set the temperature to 0
            max_tokens=500, # Set the maximum tokens to 500
            model="text-davinci-003" # Use the text-davinci-003 model
            )
        response = response["choices"][0]["text"].strip(" \n")
        response = re.sub(' {3,}', '', response)
        st.session_state.output = response
        try:
            st.session_state.search = int(re.findall('(\d{1,3})', response)[0])
        except:
            st.session_state.search = 1 
            
        #mod = st.session_state.df.loc[st.session_state.df.french == st.session_state.current_sentence].index.min()
        st.session_state.df.at[st.session_state.mod, 'last_run'] = int(datetime.utcnow().timestamp())
        st.session_state.df.at[st.session_state.mod, 'last_score'] = st.session_state.search
        st.session_state.df.to_csv(st.session_state.path, index = False)
       
st.write("Output")
st.write(st.session_state.output)
    
st.write(st.session_state.df[['french', 'last_run', 'last_score']])