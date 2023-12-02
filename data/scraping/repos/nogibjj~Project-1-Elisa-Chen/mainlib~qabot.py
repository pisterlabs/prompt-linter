# Importing necessary libraries
import os
import random
import openai

# Using dask to load and manipulate the data
from dask import dataframe as dd
from dask.distributed import Client

# Loading the data
def load_data():
    """Load the data from the csv file"""
    # Setting up the client
    client = Client()
    client.restart()

    # Loading the data
    ddf = dd.read_csv("./raw_data/reddit_questions.csv", on_bad_lines='skip', blocksize = 25e6, delimiter = ';') 
    ddf = client.persist(ddf)

    return ddf


# A Function To Find All Messages With Question Marks
def find_question_mark(df):
    """Find all messages with question marks"""
    df["question_mark"] = df["text"].str.contains(r"\?")
    df = df[df["question_mark"] == True]
    return df


# A Function to extract the question from the message
def extract_question(df):
    """Extract the question from the message"""
    df = df.copy()
    df["question"] = df['text'].str.extract(r"\b([A-Z][^.!]*[?])")[0]
    return df


# Function to generate the answer to any question using OpenAI GPT-3 API
def ask(question):
    """Generate an answer to any question using OpenAI GPT-3 API"""
    openai.api_key = os.getenv("OPENAI_API_KEY")

    start_sequence = "\nAI:"

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=question + start_sequence,
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=["\n", " Human:", " AI:"],
    )
    return response["choices"][0]["text"].strip(" \n")


# Function to randomly pick a question from the dataset and provide a response via OpenAI GPT-3 API
def answer(question):
    """Randomly pick a question from the dataset and provide a response via OpenAI GPT-3 API"""
    #warnings.filterwarnings("ignore")
    print("Question: " + question)
    answer_question = ask(question)
    print("Answer: " + answer_question)
    return answer_question



# Function to run the bot and respond to a question. By default it will respond to a random question.
def return_answer(question=""):
    """Randomly select a question from the AskReddit posts and provide an answer using OpenAI GPT-3 API"""
    if question == "":
        ddf = load_data()
        ddf = ddf.map_partitions(find_question_mark)
        ddf = ddf.map_partitions(extract_question)
        ddf = ddf.dropna(subset=["question"])
        index = random.randint(1, len(ddf))
        question = ddf["question"].compute().tolist()[index]
    bot_answer = answer(question)
    if bot_answer == '':
        bot_answer = "I don't know. Please ask me another question."
    return question, bot_answer