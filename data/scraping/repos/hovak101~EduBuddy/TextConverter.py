import os
import openai
from dotenv import load_dotenv, find_dotenv
from llama_index import VectorStoreIndex, SimpleDirectoryReader
import pyperclip

import config
load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']
MODEL = "gpt-4"

def waitAndReturnNewText():
    clipboard = pyperclip.waitForNewPaste()
    return clipboard

def translateText(text, language):
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "assistant", "content": f"You are a translator for someone only know {language} (try to translate and keep the tone and the meaning closest)."},
            {"role": "user", "content": f"Translate the following text into {language} and recognize the language: " + text},
        ],
        temperature=0
    )
    return response['choices'][0]['message']['content']

def getTitleFromText(text, language):
    response = openai.ChatCompletion.create(
        model=MODEL,

        # decide which system, assistant to use.
        # input from user data, yo uask to summarize, it will put assistant as "you are a summarizer..."

        messages=[
            {"role": "system", "content": f"You generate one very short title in {language} less than 7 words from given texts with a in the middle of the title"},
            {"role": "assistant", "content": f"You are someone that generate one title in {language} (the title should be about the text and creative)."},
            {"role": "user", "content": "Given the following text, generate one title: " + text},
        ],
        temperature=0
    )

    return response['choices'][0]['message']['content']
    
def generateSummaryFromText(text, minimumWords, maximumWords, language):
    print(language)
    response = openai.ChatCompletion.create(
        model=MODEL,

        # decide which system, assistant to use.
        # input from user data, yo uask to summarize, it will put assistant as "you are a summarizer..."
        
        messages=[
            {"role": "system", "content": f"You are a summary writer in {language} for a very busy business man so you need to be short, condense, and quick in form of bullet points."},
            {"role": "assistant", "content": f"You are someone that summarizes information in {language} on a given topic that user want to know about, make it short and condese."},
            {"role": "user", "content": "Summarize the following information in " + str(minimumWords) + " to " + str(maximumWords) + " words: " + text},
        ],
        temperature=0
    )

    return response['choices'][0]['message']['content']

def generateQuizFromText(text, numOfQuestions, language):
    response = openai.ChatCompletion.create(
        model=MODEL,

        #decide which system, assistant to use.

        messages=[
            {"role": "assistant", "content": f"You are someone that creates questions in {language} on a given topic for test user's knowledge about a given text. Question must be about the text, ask about main topic or key parts or ideas of the text"},
            {"role": "user", "content": "Create " + str(numOfQuestions) + " questions based off of the following text: " + text},
        ],
        temperature=0
    )

    return response['choices'][0]['message']['content']

def getMultipleChoiceQuiz(prompt, language, num = 5):
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": f"You are a very helpful quiz maker in {language} with this exact prompt: each line less than 100 characters of a question with 4 alternatives (1 right, 3 wrong) about {str(prompt)} formatted like this: first line: question, next four lines: alternatives. correct marked with '*' at the end of line. label alternatives 'a.'-'d.' and question '<num>.', try to make a quiz that truely test user's knowledge on the given text"},
            {"role": "assistant", "content": f"generate {str(num)} questions in {language} with 4 alternatives (1 right, 3 wrong) about {str(prompt)} formatted like this: first line: question, next four lines: alternatives. correct marked with '*' at the end of line. label alternatives 'a.'-'d.' and question '<num>.'"},
            {"role": "user", "content": "Make a" + str(num) + " question quiz about " + prompt},
        ],
        temperature=0.2  
    )
    return(response['choices'][0]['message']['content'])
    


def generateResponseFromFile(file, query):
    # Load data from a file
    documents = SimpleDirectoryReader(input_files=[file]).load_data()
    # Create an index from the loaded documents
    index = VectorStoreIndex.from_documents(documents)

    query_engine = index.as_query_engine()
    res = query_engine.query(query)
    return res

def getResponseLengthFromText(text):
    length = len(text)

    if length < 50:
        return length
    if length < 1000:
        return length // 5
    return 200; 

def translateAudio(audioFile, language):
    audio_file = open(audioFile, "rb")
    transcript = openai.Audio.translate("whisper-1", audio_file)
    return transcript.text

def sendGptRequest(prompt, context, language, memory = None):
    if not memory:
        response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a an assistant that helps user requests based on a given context. Decide what is the user's struggle or request and try to help as much as you can"},
            {"role": "assistant", "content": "You are given the following context:" + context},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        # memory = memory
        )
    else:
        response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a an assistant that helps user requests based on a given context. Decide what is the user's struggle or request and try to help as much as you can"},
            {"role": "assistant", "content": "You are given the following context:" + context},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        memory = memory
        )
    return(response['choices'][0]['message']['content'])
