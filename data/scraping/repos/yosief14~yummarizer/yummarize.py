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
import argparse
from saveRecipe import save



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
        print("No captions found for this video or the video does not exist!")
        sys.exit(1)
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
        print("Invalid youtube url")
        sys.exit(1)

# Returns the recipe from the openAI model
def getRecipeOpenAI(prompt, model, token_count):
    dotenv.load_dotenv()
    openai.api_key = os.getenv("API_KEY")
    totalCostRange = estimateCost(token_count, model) 
    requestUserAuth(totalCostRange[0], totalCostRange[1])
    completion = openai.ChatCompletion.create(model=f"{model}", messages=[{"role": "user", "content": f"{prompt}"}]) 
    return completion.choices[0].message.content

# Asks the user if they want to continue with the recipe generation given the cost
def requestUserAuth(minCost, maxCost):
    while True:
        print("The cost of this recipe is between " + str(minCost) + " and " + str(maxCost) + " cents. Do you want to continue? (y/n)")
        user_input = input()
        if user_input == "y":
            break
        elif user_input == "n":
            print("Exiting...")
            sys.exit(1)
        else:
            print("Invalid input! Please enter y or n")

def estimateCost(token_count, model):
    if model == "gpt-3.5-turbo":
        inputModelCoef = 0.0000015
        outputModelCoef = 0.000002
        maxTokens = 4000

    else:
        inputModelCoef = 0.000003
        outputModelCoef = 0.000004
        maxTokens = 16000

    minCost = (token_count * inputModelCoef)
    maxCost = ( (maxTokens - token_count)* outputModelCoef)
    
    return minCost, maxCost

def getRecipe(caption):
    query = "Summurize all of the recipes mentioned in the follwing transcript into Recipe: Ingredients: and Instructions: . For the Ingredients section and be as detailed as possible about the measurements. For the Instructions section be as detailed as possible including what is optional. If this is not a video transcript of how to cooksomething return a -1."
    context = "Transcript: \n" + caption
    prompt = query + "\n" + context 
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(caption + prompt)
    token_count = len(tokens)

    if token_count < 3000:
        return getRecipeOpenAI(prompt, "gpt-3.5-turbo", token_count)
    elif token_count >= 3000 and token_count < 15000:
        return getRecipeOpenAI(prompt, "gpt-3.5-turbo-16k", token_count)
    else:
        print("The transcript is too long to be processed by Yummarize. Please try a shorter video.")
        sys.exit(1)

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
        
    # yummarize url path/to/file, or yummarize url gdrive path/to/file
def parseArgs():
    parser = argparse.ArgumentParser(description="Yummarize is a script that summarizes youtube videos into recipe pdfs")
    parser.add_argument("url", help="The url of the youtube video you want to summarize")
    parser.add_argument("path", help="The path to the directory you want to save the recipe to. This is optional and will defualt to your home directory", default="~/", nargs='?')
    args = parser.parse_args()
    return args

def main():
    settings = parseArgs()
    user_input = settings.url
    caption = getCaptions(user_input)
    recipe = getRecipe(caption)
    videoTitle, channel = (getVideoMetaData(get_video_id(user_input)))
    save(recipe, videoTitle, channel, user_input, settings.path)


if __name__ == "__main__":
    main()

