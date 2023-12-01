from typing import List
from fastapi import APIRouter
from pydantic import BaseModel


import os
import sys
import googlemaps
from googlemaps.distance_matrix import distance_matrix

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma


from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from fastapi.responses import JSONResponse

os.environ["OPENAI_API_KEY"] = "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"

PERSIST = False

query = None


class AmbulanceLocationsModdel(BaseModel):
    id: int
    lat: str
    long: str


class AccidentLocationModdel(BaseModel):
    id: int
    lat: str
    long: str


class NearestLocationModdel(BaseModel):
    accident: AccidentLocationModdel
    ambulances: List[AmbulanceLocationsModdel]


router = APIRouter()

@router.post("/nearest-ambulance")
async def get_file(nearestLocationModdel: NearestLocationModdel):
    # Google API key
    api_key = 'AIzaSyCRpFDYJzldnE3sz5g8TGzQdNxCaQFVoiw'

    # Google Maps client
    gmaps = googlemaps.Client(key=api_key)

    nearest_location = None
    nearest_distance = float('inf')

    # Iterate through the locations and calculate distances
    for location in nearestLocationModdel.ambulances:
        # Calculate distance using the Distance Matrix API
        result = distance_matrix(gmaps, [(nearestLocationModdel.accident.lat, nearestLocationModdel.accident.long)], [
                                 (location.lat,location.long)], mode="driving")

        # Extract the distance in meters from the result
        distance = result["rows"][0]["elements"][0]["distance"]["value"]

        # Check if this location is closer than the previous closest
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_location = location

    return nearest_location,nearest_distance
