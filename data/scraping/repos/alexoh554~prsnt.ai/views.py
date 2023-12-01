from flask import Flask, jsonify, request, Response, g
from serpapi import GoogleSearch
import cohere
from flask_cors import CORS, cross_origin
from utilities import transcribe_streaming


SERP_API_KEY = "a240d146c529bc7e3c401e6ca82121c8b10f23be53d4b62537338bfbe7a11085"
COHERE_API_KEY = "OkdezIHQB1oWjY7pziAjufhf3oHXsdiPgTnb8puW"

blob_array = []


def speechblob_to_gcloud():
    blob = request.data

    # if not hasattr(g, "blob_array"):
    #     g.blob_array = []
    # g.blob_array.append(blob)
    global blob_array
    blob_array.append(blob)

    # print(g.blob_array)
    print(len(blob_array))

    if (len(blob_array) >= 10):
        # print(10)
        transcribe_streaming(blob_array)
        blob_array = []

    return jsonify({"hello": "world"})

def summarize():
    # Cohere
    co = cohere.Client(COHERE_API_KEY)

    # Get the text
    cohere_query = request.args.get("cohere_query")

    # Use Cohere to summarize the text
    response = co.summarize(
        text=cohere_query,
        length='medium',
        format='bullets',
        model='summarize-xlarge',
        additional_command='',
        temperature=0.4,
    )

    bullet_points = response[1].split('\n')

    return jsonify({
        "summary": bullet_points
    })



# Presentation page
def image_search():

    # SerpAPI endpoint
    # Searches with query and returns first google image result
    image_query = request.args.get("image_query")

    params = {
        "q": image_query,
        "engine": "google_images",
        "ijn": "0",
        "api_key": SERP_API_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    images_results = results["images_results"]

    return jsonify({
        "image": images_results[0]['original'],
    })
