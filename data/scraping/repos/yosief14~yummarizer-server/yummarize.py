#!/usr/bin/env python3
import json
import base64
import requests
import dotenv
import os
import openai
import sys
import urllib.parse as p
import urllib.request as r
import tiktoken
from flask import Flask, request as flrequest, jsonify, abort, g
import uuid
import time


# Constructs the request that graps the captions object from the video and returns it as a json object
def getCaptions(user_input):

    video_id = get_video_id(user_input)
    base64_string = base64.b64encode("\n\v{}".format(video_id).encode("utf-8")).decode("utf-8")

    headers = {
        "Content-Type": "application/json",
    }

    body = json.dumps(
        {
            "context": {"client": {"clientName": "WEB", "clientVersion": "2.9999099"}},
            "params": base64_string,
        }
    )

    response = requests.post(
        "https://www.youtube.com/youtubei/v1/get_transcript?key=AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8",
        headers=headers,
        data=body,
    ).json()


    # Parses the json object and constructs the captions input to be passed to openAI
    caption = ""
    if "actions" not in response:
        abort(400, description=f"Cannot locate captions for video with url \"{user_input}\"")
    for cueGroup in response["actions"][0]["updateEngagementPanelAction"]["content"]["transcriptRenderer"]["body"]["transcriptBodyRenderer"]["cueGroups"]:
        
        for cue in cueGroup["transcriptCueGroupRenderer"]["cues"]:
            #this is the text of the caption
            caption += cue["transcriptCueRenderer"]["cue"]["simpleText"] + "\n"

    return caption

# Parses the url and returns the video id
def get_video_id(url):
    if url.startswith("https://www.youtube.com/watch?v="):
        query = p.urlparse(url).query
        params = p.parse_qs(query)
        return params["v"][0]
    else:
        abort(400, description=f"\"{url}\" is not a valid youtube url")

def check_context_length(context):
    context_string = ""
    for message in context:
        context_string += message["content"] + "\n"
    encoding = tiktoken.get_encoding("cl100k_base")
    token_len = len(encoding.encode(context_string))
    if(token_len > 12000):
        abort(400, description=f"The transcript has a token length of {token_len} which is too long to process. Please try again with a shorter video. The maximum token length is 12,000.")
    else:
        return True

# Returns the recipe from the openAI model
def getRecipe(caption):
    dotenv.load_dotenv()
    openai.api_key = os.getenv("API_KEY")

    query = "Summurize all of the recipes mentioned in the follwing transcript into Recipe: Ingredients: and Instructions: . For the Ingredients and Instructions, Be as detailed about measurements as possible"
    context = "Transcript: \n" + caption
    content = query + "\n" + context
    system_messages=[
    {"role": "system", "content": "You are a web server designed to output JSON objects with the following format for every recipe found: {Recipe: {Ingredients: , Instructions:}} . If the transcript doesn't contain a recipe, your return value should be -1.  For the Instructions, each step should be its own value in a list. For the Ingredients each ingredient should be its own value in a list. Measurements for the Ingredients should be as detailed as possible."},
    {"role": "system", "content": "If you are having trouble, try to break down the problem into smaller parts. For example, first try to find the recipe, then try to find the ingredients, then try to find the instructions."},
    {"role": "system", "content": "The Ingredients and Instructions should be as detailed as possible. For example, if the recipe calls for 1 cup of flour, you should return 1 cup of flour, not just flour."}
    ]
    prompt = {"role": "user", "content": f"{content}"}
    system_messages.append(prompt)

    if(not check_context_length(system_messages)):
        return "The transcript is too long to process. Please try again with a shorter video."
    
    completion = openai.ChatCompletion.create(model=f"gpt-3.5-turbo-1106", response_format={"type": "json_object"}, messages=system_messages)
    return completion["choices"][0].message.content

def getVideoMetaData(video_id):
    params = {
        "format": "json",
        "url": f"https://www.youtube.com/watch?v={video_id}"
        }
    query = p.urlencode(params)
    url = "https://www.youtube.com/oembed"
    url += "?" + query

    with r.urlopen(url) as response:
        response_text = response.read()
        data = json.loads(response_text.decode())
        
    return data["title"], data["author_name"]
#Mooved the app creation to a function so that it can be used in the test file
def create_app():
    app = Flask(__name__)

    @app.route('/')
    def index():
        return 'Hello World'
    @app.before_request
    def before_request():
        execution_id = uuid.uuid4()
        g.start_time = time.time()
        g.execution_id = execution_id
        print(g.execution_id, "Route Called", flrequest.url)

    @app.route('/yummarize', methods=['GET'])
    def yummarize():
            user_input = flrequest.args.get('url')
            caption = getCaptions(user_input)
            recipe = getRecipe(caption)
            videoTitle, channel = (getVideoMetaData(get_video_id(user_input)))
            metaJson = {"title": videoTitle, "channel": channel}
            recipeJson = json.loads(recipe)
            metaJson.update(recipeJson)
            return metaJson
    return app 
app = create_app()
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)