#from sentence_transformers import SentenceTransformer
#from sklearn.metrics.pairwise import cosine_similarity
import openai
from decouple import config
import time

# Get completion
OPENAI_API_KEY = config("OPENAI_API_KEY")
openai.api_key  = OPENAI_API_KEY
def get_completion(messages, model="gpt-3.5-turbo", max_tokens=200, temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0]['message']['content']

# Get similarity between two sentences
'''
model = SentenceTransformer('bert-base-nli-mean-tokens')
def get_similarity(txt1, txt2):
    txt1_embeddings = model.encode(txt1, convert_to_tensor=True)
    txt2_embeddings = model.encode(txt2, convert_to_tensor=True)
    
    txt1_embeddings = np.array(txt1_embeddings).reshape(1, -1)
    txt2_embeddings = np.array(txt2_embeddings).reshape(1, -1)

    similarity = cosine_similarity(txt1_embeddings, txt2_embeddings)
    return similarity[0][0]
'''

# Get the duration of a task
def get_duration(start, end):
    if start[6:] == end[6:]:
        minutes = int(end[:2])*60 + int(end[3:5]) - int(start[:2])*60 - int(start[3:5])
        return minutes
    elif start[6:] == "AM" and end[6:] == "PM":
        minutes = (12 - int(start[:2]))*60 - int(start[3:5]) + int(end[:2])*60 + int(end[3:5])
        return minutes
    
def time_decrease(period):
    for i in range(1, period):
        print(f"Countdown: {period-i}", end='\r')
        time.sleep(1)
        print(" " * len(f"Countdown: {period-i}"), end='\r')