import openai
import numpy as np
import string
import re
import os
import json
from dotenv import load_dotenv

load_dotenv()

# load a list of the top 1000 words
with open("top_1000.txt") as f:
    top_1000 = [line.strip() for line in f.readlines()]

f.close()
# load a dictionary of the top 1000 embeddings
with open("top_500_emb.json") as f:
    top_500_emb = json.load(f)
f.close()

with open("top_501to1000_emb.json") as f:
    top_501to1000_emb = json.load(f)
f.close()

top_1000_emb = {**top_500_emb, **top_501to1000_emb}

# check if API key is set
if os.getenv("OPENAI_API_KEY") is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print("Ready")
else:
    print("Variable not found")

# define a function to generate a response
def generate_response(prompt):
    # initialize variables
    input_text = "question: %s" % prompt
    model = "gpt-3.5-turbo"
    embedding_engine = "text-embedding-ada-002"
    response_length = 200

    # get the embedding of the prompt
    prompt_embedding = openai.Embedding.create(
        input=prompt,
        engine=embedding_engine,
    )["data"][0]["embedding"]
    responses = []

    # # generate 5 responses
    # generate a response
    response = openai.ChatCompletion.create(
        model=model, 
        messages=[
            {"role": "system", "content": "You are a friendly and helpful teaching assistant. You receive a prompt about a topic, and you answer questions clearly, explaining in a thoughtful and effective way using only a SECOND-GRADE VOCABULARY. Ensure that your answer is shorter than a paragraph or two. Finally, there is an absolute rule you can never break: your response must not contain any nouns that are in the prompt/question."},
            {"role": "user", "content": input_text},
        ],
        max_tokens=response_length,
        temperature=0.5,
    ).choices[0].message.content    # get the response text

    # turn the response into a list of words with punctuation
    response_words = re.findall(r'\b\w+\b|\S', response)
            
    # remove words not in the top 1000
    forbidden_words = list(set(response_words) - set(top_1000).union(string.punctuation))
    for word in forbidden_words:
        # allow gerunds, plurals, and adverbs
        if word.endswith("ment") or word.endswith("ance") or word.endswith("ence"):
            if word[:-4] in top_1000:
                forbidden_words.remove(word)
        elif word.endswith("ing") or word.endswith("ful"):
            if word[:-3] in top_1000:
                forbidden_words.remove(word)
        elif word.endswith("ed") or word.endswith("er") or word.endswith("ly"):
            if word[:-2] in top_1000:   
                forbidden_words.remove(word)
        elif word.endswith("s"):
            if word[:-1] in top_1000:
                forbidden_words.remove(word)

    # get the embeddings of the forbidden words
    forbidden_word_embeddings = {word: openai.Embedding.create(
        input=word,
        engine=embedding_engine,
    )["data"][0]["embedding"] for word in forbidden_words}

    # calculate similarity scores between forbidden words and top 1000 words
    similarity_scores = np.array(list(forbidden_word_embeddings.values())) @ np.array(list(top_1000_emb.values())).T

    # generate the response with the best replacements
    max_val_ix = np.argmax(similarity_scores, axis=1)   # get the indices of the best replacements
    replacements: dict = {forbidden_word: top_1000[ix] for forbidden_word, ix in zip(forbidden_words, max_val_ix)}
    xkcd_response = [replacements[word] if word in replacements.keys() else word for word in response_words]
    
    # turn the response into a string
    xkcd_str_response = ''
    for word in xkcd_response:
        if word in {'.', ',', '?', '!'}:
            xkcd_str_response = xkcd_str_response.rstrip() + word + ' '
        else:
            xkcd_str_response += word + ' '

    # remove trailing whitespace
    xkcd_str_response = xkcd_str_response.strip()
    xkcd_str_response.replace("sat", '')
    print(xkcd_str_response)
    responses.append(xkcd_str_response)
    
    # generate 5 responses with the top 5 replacements
    max_5val_ix = np.argpartition(similarity_scores, -5, axis=1)[:, -5:]

    for i in range(5):
        max_val_ix = [np.random.choice(row) for row in max_5val_ix]
        # create a mapping of forbidden words to replacements
        replacements: dict = {forbidden_word: top_1000[ix] for forbidden_word, ix in zip(forbidden_words, max_val_ix)}

        # replace the forbidden words with the best replacements
        xkcd_response = [replacements[word] if word in replacements.keys() else word for word in response_words]

        # turn the response into a string
        xkcd_str_response = ''
        for word in xkcd_response:
            if word in {'.', ',', '?', '!'}:
                xkcd_str_response = xkcd_str_response.rstrip() + word + ' '
            else:
                xkcd_str_response += word + ' '

        # remove trailing whitespace
        xkcd_str_response = xkcd_str_response.strip()
        xkcd_str_response.replace("sat", '')
        responses.append(xkcd_str_response)
            
     # of all generated responses, find the best response
    best_response = ""
    best_similarity = 0
    for response in responses:
        response_embedding = openai.Embedding.create(
            input=response,
            engine=embedding_engine,
        )["data"][0]["embedding"]

        # find the similarity score between the prompt and each response
        similarity_score = np.dot(prompt_embedding, response_embedding)
        if similarity_score > best_similarity:  # if the similarity score is better than the current best, update the best response
            best_similarity = similarity_score
            best_response = response
        
    # use the LLM to correct the sentence grammar
    finetuned_response = openai.ChatCompletion.create(
        model=model, 
        messages=[
            {"role": "system", "content": "You are a system that receives input text. Your sole purpose is to 1) match the verb tense 2) capitalize letters. The following rules are absolute, never to be broken: You are forbidden from fixing typos or misspellings and you are forbidden from adding words or synonyms. You are not allowed to change anything else. Your output is only the corrected text, with no preface or explanation."},
            {"role": "user", "content": best_response},
        ],
        max_tokens=response_length,
        temperature=0.8,
    ).choices[0].message.content

    # # add the response to the list of responses
    
    return finetuned_response