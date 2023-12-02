import chardet
from flask import jsonify
import os
from dotenv import load_dotenv
import openai
import json
from config.db import qna_collection
from schemamodels.qna import QnAModel


load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')


# prompt generator
def generatePrompt(file, type):
    query = ""
    if type == "trasncript":
        query = f'''
You are given a transcript of a podcast. You have to generate most relevant question and answer pair for the main topic in the trasncript. Use the key concept in the transcript and frame most likely questions and answers pairs on the concept. 

Dont generate more than 10 questions, less would work depending upon the topic coverage in the transcript. 

PLEASE NOTE: GIVE THE QUESTION AND ANSWER PAIRS IN THE JSON FORMATE AS A AN ARRAY OF OBJECTS WITH THE KEYS "question" and "answer". 

The transcript is given below:
```
{file}
```
    '''
        return query
    elif type == "qna":
        query = f'''
If you are given a file data may or may not containing question and answers pairs. You have the below instructions to follow:

PLEASE FOLLOW BELOW MATCHING FORMAT

MATCHING FORMAT:
```
Question : some sample question?
Answer : answer to sample question

```
PLEASE NOTE: 
1.IF THE FILE DATA IS LIKE MATCHING FORMAT GIVE THE QUESTION AND ANSWER PAIRS IN THE JSON FORMATE AS A AN ARRAY OF OBJECTS WITH THE KEYS "question" and "answer".

2.IF YOU DO NOT GET QUESTION AND ANSWER LIKE ABOVE MATCHING FORMAT PLEASE DONT RETURN  ANYTHING. DONT EVEN CONVERT THE DATA INTO JSON.

The file containing data is below(MAY OR MAY NOT BE LIKE MATCHING FORMAT):
```
{file}

```
    '''
        return query
    else:
        return "DONT GENERATE ANYTHING"


# convert the file to json of question and answers using the openai chat api
def converToJsonviaChat(file, type, influencer_id):
    try:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        file_encoding = result['encoding']

        # Convert bytes to string using the detected encoding
        transcript = raw_data.decode(str(file_encoding))
        prompt = generatePrompt(transcript, type)
        jsonQnAs = chatGPT(prompt)

        jsonQnAs = json.loads(jsonQnAs)

        if jsonQnAs["ok"]:
            for qna in jsonQnAs["QnAs"]:
                qna = QnAModel(influencer_id=influencer_id,
                               sample_question=qna['question'], sample_answer=qna['answer'])
                qna_collection.insert_one(qna.__dict__)
            return json.dumps({"ok": True, "message": "QnAs Added"})
        else:
            return json.dumps({"ok": False, "message": str(jsonQnAs["error"])})
    except Exception as e:
        return ({"ok": False, "message": str(e)})


# connect to the openai chat api and get the response
def chatGPT(prompt):
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}]
        )
        result = completion["choices"][0]["message"]["content"]
        result = json.loads(result)
        return json.dumps({"QnAs": result, "ok": True})
    except Exception as e:
        return json.dumps({"error": str(e), "ok": False})
