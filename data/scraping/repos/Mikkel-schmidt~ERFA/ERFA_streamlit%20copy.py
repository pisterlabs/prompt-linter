import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import tiktoken

import openai

st.title('FagBotten')

openai.api_key = st.secrets["apikey"]

if 'df' not in st.session_state:
    df = pd.read_csv('df_enc.csv')
    st.session_state['key'] = df

with open('document_embeddings.pkl', 'rb') as fp:
    document_embeddings = pickle.load(fp)

EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETIONS_MODEL = "gpt-3.5-turbo"

## This code was written by OpenAI: https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb

def get_embedding(text: str, model: str="text-embedding-ada-002") -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.content) for idx, r in df.iterrows()
    }

#document_embeddings = compute_doc_embeddings(df)

## This code was written by OpenAI: https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb


def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_by_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities
    
MAX_SECTION_LEN = 2000
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_by_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    st.write(f"Vigtigste {len(chosen_sections)} kilder:")
    #st.write("\n".join(chosen_sections))
    st.write(df['Kilde'].iloc[chosen_sections_indexes].values)
        
    return chosen_sections, chosen_sections_len

def answer_with_gpt_4(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False
) -> str:
    messages = [
        {"role" : "system", "content":"Du er en rådgiver chatbot der kun kan svare ud fra den kontekst du er blevet tilført her. Hvis du ikke kan svare på spørgsmålet skal du svare 'Svaret er ikke i ERFA bladene, håndbogen eller Sikkerhedsstyrelsens guider.'"}
    ]
    prompt, section_lenght = construct_prompt(
        query,
        document_embeddings,
        df
    )
    if show_prompt:
        print(prompt)


    context= ""
    for article in prompt:
        context = context + article 


    context = context + '\n\n --- \n\n + ' + query

    messages.append({"role" : "user", "content":context})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=2000,
        messages=messages
        )

    return '\n' + response['choices'][0]['message']['content'], section_lenght

prompt = st.text_input('Indtast spørgsmål til ERFA-bladene, sikkerhedsstyrelsens guider eller håndbogen:', )
if prompt:
    c = st.container()
    response, sections_tokens = answer_with_gpt_4(prompt, df, document_embeddings)
    c.write(response)


    st.link_button('Kilder', 'https://github.com/Mikkel-schmidt/ERFA/tree/main/Docs')




#st.write(df)