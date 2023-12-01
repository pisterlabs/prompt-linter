from flask import Flask, jsonify
from flask import request
import whisper
import pymongo
from bson.objectid import ObjectId
import os
import openai
import json
from dotenv import load_dotenv
import cohere
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
import requests

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")

client = pymongo.MongoClient("localhost", 27017)
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
db = client["suturelogs_db_1"]
collection = db["surgeries"]


app = Flask(__name__)


def extract_json_substring(string):
    # Find the first occurrence of a curly brace '{' or square bracket '['
    if string.find('{') == -1 and string.find('[') == -1:
        return ''
    if string.find('{') == -1:
        start_index = string.find('[')
    elif string.find('[') == -1:
        start_index = string.find('{')
    else:
        start_index = min(string.find('{'), string.find('['))
    print(start_index)
    if start_index == -1:
        # No opening brace or bracket found, return None
        return None

    # Track the number of opening and closing braces/brackets encountered
    count = 1
    end_index = start_index + 1

    while end_index < len(string):
        if string[end_index] == '{' or string[end_index] == '[':
            count += 1
        elif string[end_index] == '}' or string[end_index] == ']':
            count -= 1

        if count == 0:
            # Found matching closing brace or bracket, extract the JSON substring
            json_substring = string[start_index:end_index + 1]

            try:
                # Validate if the substring is a valid JSON
                json.loads(json_substring)
                return json_substring
            except json.JSONDecodeError:
                # Invalid JSON, continue searching for another substring
                pass

        end_index += 1

    # No valid JSON substring found, return None
    return ''


@app.route('/transcribe', methods=['POST'])
def read_file():
    try:
        # get value from form body
        data = request.get_json()
        audioPath = os.getenv("AUDIO_PATH") + \
            data['audioPath']+".wav"
        model = whisper.load_model("medium.en", "cpu")
        result = model.transcribe(audioPath)
        val = []
        for i in result['segments']:
            segment = {
                "start": i['start'],
                "end": i['end'],
                "text": i['text']
            }
            val.append(segment)

        starts = []
        text = []
        for item in val:
            starts.append(round(item['start']))
            text.append(item['text'].strip())

        collection.update_one({"_id": ObjectId(data['surgeryId'])}, {
            "$set": {
                "videoTimestamps": starts,
                "transcript": text,
            }
        })

        transcript = ""
        for i in text:
            transcript = transcript + i + " "
        url = "https://api.edenai.run/v2/text/summarize"

        payload = {
            "response_as_dict": True,
            "attributes_as_list": False,
            "show_original_response": False,
            "output_sentences": 4,
            "providers": "microsoft",
            "text": transcript,
            "language": "en"
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": "Bearer " + os.getenv("EDEN_API_KEY")
        }

        edenres = requests.post(url, json=payload, headers=headers)
        m = edenres.json()
        summary = m['microsoft']['result']
        collection.update_one({"_id": ObjectId(data['surgeryId'])}, {
            "$set": {
                "summary": summary

            }
        })

        sectionsPrompt = "Transcript: "+str(text) + "\n" + \
            "Video Timestamps: "+str(starts) + "\n" + \
            ''' Given is the transcript and the timestamps which have the starting time of the transcript in seconds. 
                Command: Generate important sections as titles from the transcript it's its start time and end time according to the timestamps. Your output must be a valid JSON, and adhere to the Format and the Rules.
                Format :  [{"title" : "section title", "startTime" : 20, "endTime" : 40 }] 
                Rules: 
                1. JSON must not be stringified.
                2. Your response must not have anything else apart from the JSON.
            '''

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=sectionsPrompt,
            temperature=1,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        # sections = json.loads(response['choices'][0]['text'])
        try:
            sectionsText = extract_json_substring(
                response['choices'][0]['text'])
            sections = json.loads(sectionsText)
        except:
            print("Could not generate sections")
            sections = json.loads([])

        # surgery = collection.find_one({"_id": ObjectId(data['surgeryId'])})
        # toupdate = {
        #     "videoTimestamps": starts,
        #     "transcript": text,
        #     "sectionsInVideo": sections
        # }
        # surgery.update(toupdate)
        # collection.replace_one({"_id": ObjectId(data['surgeryId'])}, surgery)
        collection.update_one({"_id": ObjectId(data['surgeryId'])}, {
            "$set": {
                "sectionsInVideo": sections,
            }
        })
        return "success"
    except:
        return "error"


@app.route('/semantic-search', methods=['POST'])
def search():
    try:
        co = cohere.Client(cohere_api_key)
        data = request.get_json()
        searchquery = data['searchquery']
        surgeries = data['dbsurgeries']
        surgericalTranscripts = []
        for document in surgeries:
            transcriptArray = document['transcript']
            transcript = "Title : " + document['surgeryTitle'] + " "
            for i in transcriptArray:
                transcript = transcript + i + " "
            surgericalTranscripts.append(transcript)
        print(surgericalTranscripts)
        print(surgeries)
        df = pd.DataFrame({'transcript': surgericalTranscripts})
        embeds = co.embed(texts=list(df["transcript"]),
                          model="large",
                          truncate="RIGHT").embeddings
        embeds = np.array(embeds)
        search_index = AnnoyIndex(embeds.shape[1], 'angular')
        for i in range(len(embeds)):
            search_index.add_item(i, embeds[i])
        search_index.build(10)

        query_embed = co.embed(texts=[searchquery],
                               model="large",
                               truncate="RIGHT").embeddings
        similar_item_ids = search_index.get_nns_by_vector(
            query_embed[0], 10, include_distances=True)
        print(similar_item_ids)
        results = similar_item_ids[0]
        ress = []
        for i in range(len(results)):
            ress.append(surgeries[results[i]])
        return jsonify({"status": "success", "surgeries": ress})
    except:
        return jsonify({"status": "error"})


if __name__ == "__main__":
    app.run()
