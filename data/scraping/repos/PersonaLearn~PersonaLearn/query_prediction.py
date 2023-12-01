#!/usr/bin/env python3

import os
from dotenv import load_dotenv
import openai

from backend.models.query_predictor.index import predict_query_from_transcript

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
openai.api_key = os.getenv("OPENAI_API_KEY")

while True:
    title = input("Video Title: ")
    transcript = input("Section Transcript: ")
    query = predict_query_from_transcript(title, transcript)
    print(f"Video query: {query}")
