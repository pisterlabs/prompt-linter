import os
import numpy as np
import pandas as pd
import typing as t
import sklearn
import gensim
from sklearn.neighbors import NearestNeighbors
import openai
import re

word_vectors = gensim.models.KeyedVectors.load_word2vec_format('qna/word_vectors.bin', binary=True)
question_embeddings = pd.read_csv('qna/question_embeddings.csv')
feedback_embeddings = pd.read_csv('qna/feedback_embeddings.csv')
question_docs = pd.read_csv('qna/question_dataframe.csv')
feedback_docs = pd.read_csv('qna/feedback_dataframe.csv')

openai.api_key = os.environ['OPENAI_API_KEY']

def get_questions(question_string):
    questions = question_string.strip().split("\n")
    result = []
    for question in questions:
        if any(char.isdigit() for char in question):
            result.append(question.strip())
    return result

# Returns array of questions

def api_get_question(job_description: str, experience_level: str, number_of_questions: int, content:str):
 
    prompt = f"Write questions related to this job descripition: '{job_description}', and that most fit {experience_level}. Put them in a list with this format:\nQuestion 1: \nQuestion 2:\nQuestion 3: etc...\nUse the content below to inspire you. Your response should contain only questions. There should be {number_of_questions}. IF the content below doesn't seem relevant to the job description, create questions that are. \nContent (pick {number_of_questions} question(s) most relevant to the job description, and as comprehensive as possible given the number of questions):\n{content}\n [END OF CONTENT] DO NOT ANSWER THESE QUESTIONS YOURSELF! Write questions related to this job descripition: '{job_description}', and that most fit {experience_level}. Put them in a list with this format:\nQuestion 1: \nQuestion 2:\nQuestion 3: etc..."

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.7,
    )

    questions = response.choices[0].text.strip()
    # print('HERE ARE THE GENERATED INTERVIEW QUESTIONS:', questions)
    
    return response, get_questions(questions)

def api_get_feedback(question: str, user_response: str, job_description):
 

    # question = 'Question 1: What are the different types of Machine Learning algorithms?'
    # user_response = "knn and neural networks"
    # job_description = 'Machine Learning'

    prompt = f"Act like you are giving feedback on a job interview, and are helping the person being interviewed improve. Based on this questions:  {question}\nAnd given this response: {user_response}\nFor this job: {job_description}\nGive constructive feedback for the response based on the content below. If you find the user's response to be a good answer the question, let them know and why. Otherwise, tell them how they could do better:\n[RELEVANT CONTENT]"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )

    feedback= response.choices[0].text.strip()
    print(feedback)

    
    return feedback

# feedback_docs =
# feedback_embeddings = 


INDEX_NAME = "embedding-index"
NUM_VECTORS = 4000
PREFIX = "embedding"
VECTOR_DIM = 1536
DISTANCE_METRIC = "COSINE"



def get_embeddings(text: str):
    df = pd.DataFrame(columns=['title', 'heading', 'content', 'tokens'])
    title = 'query'
    heading = 'query'
    content = text
    tokens = content.split()


    df.loc[len(df)] = [title, heading, content, tokens]

    # Generate embeddings for each row in the DataFrame
    embeddings = []
    for index, row in df.iterrows():
        text = row['content']  # the text column name in your CSV file
        words = text.split()
        vectors = []
        for word in words:
            try:
                vector = word_vectors[word]
                vectors.append(vector)
            except KeyError:
                # Ignore words that are not in the pre-trained word embeddings
                pass
        if vectors:
            # Calculate the mean of the word embeddings to get a document embedding
            doc_embedding = np.mean(vectors, axis=0)
            embeddings.append(doc_embedding)
        else:
            # Use a zero vector if none of the words are in the pre-trained word embeddings
            embeddings.append(np.zeros(100))

    # Add the embeddings as new columns in the DataFrame
    for i in range(100):
        df[i] = [embedding[i] for embedding in embeddings]

    # Save the DataFrame with the embeddings to a new CSV file
    return df

# Load documents and their embeddings
question_embeddings_df = question_embeddings.drop(columns=["title", "heading", "content"])
question_embeddings_arr = question_embeddings_df.to_numpy()
question_knn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
question_knn_model.fit(question_embeddings_arr)

feedback_embeddings_df = feedback_embeddings.drop(columns=["title", "heading", "content"])
feedback_embeddings_arr = feedback_embeddings_df.to_numpy()
feedback_knn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
feedback_knn_model.fit(feedback_embeddings_arr)



def get_most_relevant(is_questions: bool, text: str):
   
    # Load documents and their embeddings
    if is_questions:
        docs_df = question_docs 
    else:
        docs_df = feedback_docs
        
    # Load embedding of user query
    query_embedding = get_embeddings(text)
    query_embedding = query_embedding.drop(columns=["title", "heading", "content","tokens"]) # Drop the 'title' column
    query_embedding = query_embedding.to_numpy() # Convert to numpy array

    # Find the indices of the nearest neighbors to the query
    if is_questions:
        indices = question_knn_model.kneighbors(query_embedding, return_distance=False)
    else:
        indices = feedback_knn_model.kneighbors(query_embedding, return_distance=False)


    # Get the documents corresponding to the nearest neighbors
    top_5_knn_docs = docs_df.iloc[indices[0]]
    return top_5_knn_docs
    
def get_content_as_string(top_5: pd.DataFrame):

    top_5['content']
    content_string = '//\n'.join(top_5['content'])
    # Replace the newline characters ("\n") with a new line ("\n") and double slashes ("//")
    content_string = content_string.replace('\n', '\n')
    content_string = content_string[:3900]
    return (content_string)

def api_get_final_feedback(feedback, job_description):
 


    prompt = f"Act like you are giving feedback on a job interview, and are helping the person being interviewed improve. Based on all this feedback you gave:  {feedback}\nFor this job: {job_description}\nGive the person being inverviewd an overall score out of 100, then an overall summary on how they did."
    response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.5,
        )

    final = response.choices[0].text.strip()

    # Extract the score from the final string
    score_match = re.search(r"\d+", final)
    score = int(score_match.group(0)) if score_match else None
    
    return final, score