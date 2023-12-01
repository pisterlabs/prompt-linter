from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response
import openai as OpenAI
import pinecone
from plexapi.server import PlexServer
from enum import Enum
from src.core.config import settings
from typing import Any, Dict, Tuple
from src.api.deps import get_redis_client
from src.db.vector import PineconeDB, get_openai, create_embeddings
import hashlib
import logging
import json

# Setting up logging
logger = logging.getLogger(__name__)


router = APIRouter()

def get_plex():
    return PlexServer(settings.PLEX_URL, settings.PLEX_TOKEN)

def generate_metadata_description(item) -> Tuple[dict, str]:
    description_elements = []
    metadata = {}

    if item.title: 
        description_elements.append(f"The {item.TYPE}: {item.title}")
        metadata["TYPE"] = item.TYPE
        metadata["title"] = item.title

    if item.year:
        description_elements.append(f"released in {item.year}")
        metadata["year"] = item.year

    if item.genres:
        genres = ', '.join([genre.tag for genre in item.genres])
        description_elements.append(f"is in genres {genres}")
        metadata["genres"] = genres

    if item.actors:
        actors = ', '.join([role.tag for role in item.actors])
        description_elements.append(f"It features actors {actors}")
        metadata["actors"] = actors

    if hasattr(item, 'contentRating') and item.contentRating:
        description_elements.append(f"is a {item.contentRating}-rated")
        metadata["contentRating"] = item.contentRating

    if hasattr(item, 'studio') and item.studio:
        description_elements.append(f"Produced by {item.studio}")
        metadata["studio"] = item.studio

    if hasattr(item, 'audienceRating') and item.audienceRating is not None:
        ratescore = item.audienceRating
        if ratescore >= 8:
            sentiment = "highly rated"
        elif ratescore >= 5:
            sentiment = "moderately rated"
        else:
            sentiment = "low rated"
        description_elements.append(f"This {item.TYPE} is {sentiment} by the audience with a rating score of {ratescore}")
        metadata["audienceRating"] = ratescore

    if item.summary:
        description_elements.append(f"Summary: {item.summary}")
        metadata["summary"] = item.summary

    description = ', '.join(description_elements)

    return metadata, description

def generate_description_from_metadata(metadata):
    description_elements = []

    if "TYPE" in metadata and "title" in metadata:
        description_elements.append(f"The {metadata['TYPE']}: {metadata['title']}")

    if "year" in metadata:
        description_elements.append(f"released in {metadata['year']}")

    if "genres" in metadata:
        description_elements.append(f"is in genres {metadata['genres']}")

    if "actors" in metadata:
        description_elements.append(f"It features actors {metadata['actors']}")

    if "contentRating" in metadata:
        description_elements.append(f"is a {metadata['contentRating']}-rated")

    if "studio" in metadata:
        description_elements.append(f"Produced by {metadata['studio']}")

    if "audienceRating" in metadata and metadata["audienceRating"] is not None:
        ratescore = metadata['audienceRating']
        sentiment = "highly rated" if ratescore >= 8 else "moderately rated" if ratescore >= 5 else "low rated"
        description_elements.append(f"This {metadata['TYPE']} is {sentiment} by the audience with a rating score of {ratescore}")

    if "summary" in metadata:
        description_elements.append(f"Summary: {metadata['summary']}")

    description = ', '.join(description_elements)

    return description
@router.get("/index-plex", tags=[" "])
async def sync_db():

    vector_db = PineconeDB("media-index", dimension=1536, metric='cosine', shards=1)
    redis = await get_redis_client()
    plex = get_plex()
    
    descriptions = []
    items_for_embedding = []

    for section_name in ['Movies', 'TV Shows']:
        for item in plex.library.section(section_name).all():
            metadata, description = generate_metadata_description(item)
            # create a hash of the metadata
            metadata_hash = hashlib.sha256(str(metadata).encode()).hexdigest()
            old_hash = await redis.get(item.ratingKey)
            if old_hash and old_hash == metadata_hash:
                continue
            items_for_embedding.append((metadata, str(item.ratingKey), metadata_hash))
            descriptions.append(description)
    if not descriptions:
        logger.info("No new items to index.")
        return Response(status_code=204)
    embeddings = create_embeddings(descriptions)
    vectors = [pinecone.Vector(id=ratingKey, values=embeddings[idx], metadata=metadata) for idx, (metadata, ratingKey,_) in enumerate(items_for_embedding)]
                
    if vectors:
        try:
            logger.info("Starting to upsert vectors...")
            vector_db.upsert(vectors=vectors, batch_size=100)
            await redis.mset({ratingKey: metadata_hash for _, ratingKey, metadata_hash in items_for_embedding})
            logger.info("Upsert operation successful.")
        except Exception as e:
            logger.error(f"Unable to upsert vectors: {str(e)}")
    return Response(status_code=200)


class ModelName(str, Enum):
    gpt_4 = "gpt-4-0613"
    gpt_3 = "gpt-3.5-turbo-0613"
    # gpt_3_16k = "gpt-3.5-turbo-16k"
    

def vector_query(query,filter=None):
    """Function to query Pinecone DB"""
    vector_db = PineconeDB("media-index", dimension=1536, metric='cosine', shards=1)
    res = vector_db.query(query=query, top_k=5, filter=filter)
    return '/n'.join([generate_description_from_metadata(item.metadata) for item in res])

@router.get("/querydb", tags=["Queries"])
async def query_db(query: str, model: ModelName = ModelName.gpt_3) -> Dict[str, Any]:
    functions = [
        {
            "name": "vector_query",
            "description": "Query the Pinecone vector DB for movies and shows currently available on plex. You should filter out irrelevant results yourself before answering the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to be searched against the index",
                    },
                    "filter":{
                       "type": "object",
                            "properties": {
                                "TYPE": {
                                    "type": "string",
                                    "enum": ["movie", "show"],
                                    "description": "Type of media to filter by"
                                }
                            }
                        
                    }
                },
                "required": ["query", "filter"],
            }
            
        }
    ]
    
    # Step 1: send the conversation and available functions to GPT
    messages = [
        {"role":"system", "content": "You are the plex assistant that can help users to find movies and shows on a personal plex instance. When calling the vector_query function adapt the query to represent the user's sentiment to provide better similarity matching. Choose the best resutls to return to"},
        {"role": "user", "content": query }]
    response = OpenAI.ChatCompletion.create(
        model=model,
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    response_message = response["choices"][0]["message"]
    
    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        available_functions = {
            "vector_query": vector_query,
        }
        function_name = response_message["function_call"]["name"]
        function_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = str(function_to_call(
            **function_args
        ))
        messages.append(response_message)  # extend conversation with assistant's reply
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response
        second_response = OpenAI.ChatCompletion.create(
            model=model,
            messages=messages,
        )  # get a new response from GPT where it can see the function response
        return second_response
