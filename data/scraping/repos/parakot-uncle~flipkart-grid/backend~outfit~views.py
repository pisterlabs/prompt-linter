import json
import os
import re

import pandas as pd
import requests
from django.http import JsonResponse
from dotenv import load_dotenv
from langchain.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from PIL import Image
from rest_framework import generics
from rest_framework.response import Response
from rest_framework.views import APIView
from vton.pipe import runPipeline

from .models import Outfit, Outfit_Accessory, Outfit_Occasion
from .serializers import (OutfitAccessorySerializer, OutfitOccasionSerializer,
                          OutfitSerializer)

load_dotenv()

item_data = os.getenv("ITEM_DATA_CSV")
mapped_outfits = os.getenv("MAPPED_OUTFITS_CSV")

llm = OpenAI(temperature=0)

prompt_agent = create_csv_agent(
    llm,
    item_data,
    verbose=True,
)

recommender_agent = create_csv_agent(llm, mapped_outfits, verbose=True)

output_parser = CommaSeparatedListOutputParser()

fashion_df = pd.read_csv(item_data)

pattern = r"\b\d{4,5}\b"  # Match 4 to 5 digits


def extract_ids(arr):
    extracted_ids = []

    for item in arr:
        matches = re.findall(pattern, item)
        extracted_ids.extend(matches)

    return extracted_ids


def get_item_details_from_id(item_id):
    try:
        id = int(item_id)
        row = fashion_df[fashion_df["id"] == id]
        image = row["link"].values[0]
        category = row["subCategory"].values[0].lower()
        name = row["productDisplayName"].values[0]
        gender = row["gender"].values[0]

        res = dict()
        res["id"] = id
        res["image"] = image
        res["category"] = category
        res["name"] = name
        res["gender"] = gender

        return res
    except:
        return dict()


class OutfitSelection(APIView):
    def post(self, request, format=None):
        categories = ["topwear", "bottomwear", "shoes"]

        selected_items = request.data.get("selected_items")
        keys = list(selected_items.keys())
        for category in selected_items.keys():
            if category in categories:
                categories.pop(categories.index(category))

        if len(categories) == 1:
            item1 = keys[0]
            item2 = keys[1]
            query = f"Get 3 unique {category[0]} where {item1} = {selected_items[item1]} and {item2} = {selected_items[item2]}"
            result = output_parser.parse(recommender_agent.run(query))
            extracted_result = extract_ids(result)
            res = []
            for id in extracted_result:
                res.append(get_item_details_from_id(id))
            return JsonResponse({str(categories[0]): res})

        item1 = keys[0]
        result = []
        query = f"Get 3 unique {categories[0]} where {item1} = {selected_items[item1]}"
        result1 = output_parser.parse(recommender_agent.run(query))
        extracted_result1 = extract_ids(result1)
        res1 = []

        for id in extracted_result1:
            res1.append(get_item_details_from_id(id))

        query = f"Get 2 unique {categories[1]} where {item1} = {selected_items[item1]} and {categories[0]} = "
        for i in range(len(extracted_result1)):
            query += str(extracted_result1[i]) + " "
            if i < len(extracted_result1) - 1:
                query += "or "
        print("query", query)
        result2 = output_parser.parse(recommender_agent.run(query))
        extracted_result2 = extract_ids(result2)
        res2 = []

        for id in extracted_result2:
            res2.append(get_item_details_from_id(id))
        return JsonResponse({str(categories[0]): res1, str(categories[1]): res2})


class OutfitPromptList(APIView):
    def post(self, request, format=None):
        query = (
            f"{request.data.get('query')} for {request.data.get('gender')} (Get 5 ids)"
        )

        previous_queries = request.data.get("previous_queries")
        if len(previous_queries) > 0:
            query + "Extract information from previous queries - "
            for prev_query in previous_queries:
                query += prev_query

        result = output_parser.parse(prompt_agent.run(query))
        res = []

        for id in result:
            res.append(get_item_details_from_id(id))

        return JsonResponse({"results": res})


class OutfitList(generics.ListCreateAPIView):
    serializer_class = OutfitSerializer

    def get(self, request, *args, **kwargs):
        result_set = Outfit.objects.filter(user=self.request.query_params.get("user"))
        res = []
        for obj in list(result_set):
            ans = {
                "topwear": get_item_details_from_id(obj["topwear"]),
                "bottomwear": get_item_details_from_id(obj["topwear"]),
                "shoes": get_item_details_from_id(obj["topwear"])
            }
            res.append(ans)

        return JsonResponse({"outfits": res})


class OutfitPreview(generics.RetrieveAPIView):
    def get(self, request, *args, **kwargs):
        id = self.kwargs["pk"]
        item = get_item_details_from_id(id)

        data = requests.get(item["image"]).content
        print(os.getcwd())

        current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path to the 'cloth_image' directory
        cloth_image_relative_path = os.path.join(current_directory, '..', 'data', 'cloth_image')

        f = open(f"{cloth_image_relative_path}/{id}.jpg",'wb')
  
        f.write(data)
        f.close()

        runPipeline(id)