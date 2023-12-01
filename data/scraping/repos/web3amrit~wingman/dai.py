import os
import openai
import asyncio
import aioconsole
import logging
from aiohttp import ClientSession
from fastapi import FastAPI, HTTPException
from typing import List
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import (
    ComplexField,
    CorsOptions,
    SearchIndex,
    ScoringProfile,
    SearchFieldDataType,
    SimpleField,
    SearchableField
)

from prompting import system_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

preset_questions = [
    "What is her age range?",
    "What clothes is she wearing?",
    "What activity is she involved in",
    "Is there anything else I need to know?"
]

async def async_input(prompt: str = "") -> str:
    try:
        return await aioconsole.ainput(prompt)
    except Exception as e:
        logger.error(f"Error during input: {str(e)}")
        return ""

search_service_name = os.getenv('search_service_name')
endpoint = "https://wingmanindexing.search.windows.net"
admin_key = os.getenv('admin_key')
search_index_name = "wingmanindexdai"
index_name = "wingmanindexdai"
openai_api_key = os.getenv('openai_api_key')

search_client = SearchClient(endpoint=endpoint,
                             index_name=search_index_name,
                             credential=AzureKeyCredential(admin_key))

async def search_chunks(query):
    try:
        results = search_client.search(search_text=query)
        return [result for result in results]
    except Exception as e:
        logger.error(f"Error searching chunks: {str(e)}")
        return []

async def insert_chunks(chunks):
    try:
        actions = [{"@search.action": "upload", "id": str(i+1), "content": chunk} for i, chunk in enumerate(chunks)]
        search_client.upload_documents(documents=actions)
    except Exception as e:
        logger.error(f"Error inserting chunks: {str(e)}")
        return []

@app.post("/uploadedimage")
async def receive_uploaded_image(image: dict):
    try:
        image_url = image.get('image_url', '')
        image_name = image.get('file_name', '')
        if not image_url or not image_name:
            raise ValueError("Invalid image data received.")

        image_description_task = asyncio.create_task(describe_image(image_url))
        questions_task = asyncio.create_task(ask_preset_questions())

        image_description = await image_description_task
        situation, history = await questions_task

        history.append({"role": "assistant", "content": "I see the following in the image: " + image_description})
        situation = image_description + " " + situation

        return {"situation": situation, "history": history}
    except Exception as e:
        logger.error(f"Error receiving uploaded image: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred. Please try again later.")

async def describe_image(image_url):
    try:
        subscription_key = os.environ['COMPUTER_VISION_KEY']
        endpoint = os.environ['COMPUTER_VISION_ENDPOINT']
        computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

        loop = asyncio.get_running_loop()

        analysis = await loop.run_in_executor(None, lambda: computervision_client.analyze_image(
            image_url,
            visual_features=[
                VisualFeatureTypes.objects,
                VisualFeatureTypes.tags,
                VisualFeatureTypes.description,
                VisualFeatureTypes.color,
            ]
        ))
        if (len(analysis.description.captions) == 0):
            image_description = "No description detected."
        else:
            image_description = analysis.description.captions[0].text

        objects = [obj.object_property for obj in analysis.objects]
        tags = [tag.name for tag in analysis.tags]
        dominant_colors = analysis.color.dominant_colors

        return f"I see the following in the image: {image_description}. The image contains these objects: {', '.join(objects)}. The image has these dominant colors: {', '.join(dominant_colors)}."

    except Exception as e:
        logger.error(f"Error describing image: {str(e)}")
        return "Unable to process image."

async def ask_preset_questions():
    history = []
    situation = ""

    for question in preset_questions:
        logger.info(question)
        history.append({"role": "assistant", "content": question})
        answer = await async_input()
        history.append({"role": "user", "content": answer})
        situation += answer + " "

    return situation, history

def select_top_pickup_lines(response, num_lines):
    pickup_lines = []
    for choice in response['choices']:
        line = choice['message']['content']
        pickup_lines.append(line)
        if len(pickup_lines) >= num_lines:
            break
    return pickup_lines

async def generate_pickup_lines(situation, history, num_lines):
    relevant_data = await search_chunks(situation)
    for data in relevant_data:
        situation += data[1] + " "

    messages = history + [
        {
            "role": "assistant",
            "content": f"{system_message}\nSituation: {situation}\nGenerate {num_lines} pickup lines:"
        }
    ]

    retry_attempts = 3
    retry_delay = 2

    for attempt in range(retry_attempts):
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.15,
                n=1,
            )

            pickup_lines = select_top_pickup_lines(response, num_lines)

            for line in pickup_lines:
                history.append({"role": "assistant", "content": line})

            return pickup_lines

        except openai.error.RateLimitError as e:
            logger.error(f"Rate limit exceeded: {str(e)}")
            await asyncio.sleep(retry_delay)
        except Exception as e:
            logger.error(f"Error generating pickup lines: {str(e)}")
            return ["Error generating pickup lines."]

async def process_user_query(query, history):
    try:
        history.append({"role": "user", "content": query})
        relevant_chunks = await search_chunks(query)
        openai.api_key = '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'
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=history + [{"role": "user", "content": query}],
            max_tokens=200,
            n=1,
            temperature=0.7
        )

        assistant_message = response['choices'][0]['message']['content']
        history.append({"role": "assistant", "content": assistant_message})

        return assistant_message
    except openai.error.RateLimitError as e:
        logger.error(f"Rate limit exceeded: {str(e)}")
        return "Rate limit exceeded."
    except Exception as e:
        logger.error(f"Error processing user query: {str(e)}")
        return "Unable to process query."

def save_history_to_file(history):
    try:
        with open('history.txt', 'w') as file:
            for item in history:
                file.write(f"{item['role']}: {item['content']}\n")
    except Exception as e:
        logger.error(f"Error saving history to file: {str(e)}")

async def main():
    try:
        situation, history = await ask_preset_questions()
        pickup_lines = await generate_pickup_lines(situation, history, 5)
        logger.info("\n".join(pickup_lines))
        while True:
            query = await async_input("\nEnter your query: ")
            response = await process_user_query(query, history)
            logger.info(response)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user. Exiting.")
        save_history_to_file(history)
        exit()

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")