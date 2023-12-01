from bs4 import BeautifulSoup
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, WebSocket
import json

import asyncio
import os
import adopenai_utils
import pexels_utils
import advideo_utils
import video_utils
from fastapi import Body
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from uuid import uuid4
from moviepy.audio.AudioClip import concatenate_audioclips
from moviepy.video.io.VideoFileClip import VideoFileClip
from pyppeteer import launch
import soundfile as sf
from moviepy.editor import ImageClip, concatenate_videoclips,AudioFileClip , CompositeVideoClip
import os
import requests
import openai
from elevenlabs import set_api_key,generate
import ast
from typing import List
from pymongo import MongoClient
from datetime import datetime
import aiofiles
import sys
print(sys.path)

MONGO_URL = 
client = MongoClient(MONGO_URL)
db = client["test"]


openai.api_key = 


set_api_key()

ads_collection = db['ads']
presentations_collection = db['presentations']


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Constants for Azure Storage
AZURE_STORAGE_CONNECTION_STRING = 
CONTAINER_NAME = 'mosivideo'



async def screenshot(url, delay, num_screenshots):
    browser = await launch()
    pages = await browser.pages()
    page = pages[0] if pages else await browser.newPage()

    try:
        await page.goto(url, timeout=60_000)
    except Exception as e:
        print(f"Timed out with error: {e}")
        await browser.close()
        return []

    await asyncio.sleep(delay)

    screenshot_paths = []
    for i in range(num_screenshots):
        screenshot_path = f"screenshot_{i}.png"
        screenshot_paths.append(screenshot_path)
        await page.screenshot({'path': screenshot_path})
        await page.evaluate("window.scrollBy(0, window.innerHeight);")
        await asyncio.sleep(delay)

    await browser.close()
    return screenshot_paths


def images_to_transition_video(screenshot_paths, audio_paths, video_name, fps, durations):
    if not screenshot_paths:
        print("No screenshots to process.")
        return

    if len(screenshot_paths) != len(audio_paths):
        print("The number of images and audio files do not match.")
        return

    clips = []
    for img_path, audio_path, duration in zip(screenshot_paths, audio_paths, durations):
        audio_clip = AudioFileClip(audio_path)
        clip = ImageClip(img_path).set_duration(duration)
        final_clip = CompositeVideoClip([clip.set_audio(audio_clip)])
        clips.append(final_clip)

    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(video_name, fps=fps)



@app.post("/createad")
async def create_ad_video(prompt: str = Body(...)):

    unique_id = uuid4().hex
    temp_html_filename = f'temp_ad_{unique_id}.html'
    final_video_filename = f'final_ad_video_{unique_id}.mp4'
    def get_slides_content():
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "You are a content creator for a 15-second video advertisement of only 3 slides. Provide content in the format of an array of objects, where each object represents a slide with a 'title' and 'description' and 'script'. Example: [{'title': 'Title 1', 'description': 'Description 1', 'script':'Script 1'}, {'title': 'Title 2', 'description': 'Description 2', 'script':'Script 2'}]. The script is the text that wil be read out in the ad , what the sales message is and should end with the call to action like Call us now or Email us now or Check out our store or check out our site today and Please make the scripts short and to the point use only one short sentence for each slide"},
                    {"role": "user", "content": prompt},
                ]
            )
            print(response["choices"][0]["message"]["content"])
            slides_content = ast.literal_eval(response.choices[0].message.content)
            return slides_content

        except Exception as e:
            print(f"Error fetching content from OpenAI: {e}")
            return []

    slides_content = get_slides_content()

    def get_images_from_pexels(query, api_key, count=4):
        url = f"https://api.pexels.com/v1/search?query={query}&per_page={count}"
        headers = {"Authorization": api_key}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            print("Images data from Pexels:", response.content)
            return [photo['src']['medium'] for photo in data['photos']]
        else:
            print(f"Error {response.status_code} from Pexels.")
            return []

    API_KEY = '563492ad6f91700001000001f962fb126b81404f9e422da0fae33f29'  # Redacted for security
    query = "men"
    image_urls = get_images_from_pexels(query, API_KEY, 3)

    with open('ad.html', 'r') as file:
        original_html_content = file.read()

    if len(image_urls) >= 2 and len(slides_content) >= 3:
        temp_html_content = original_html_content
        temp_html_content = temp_html_content.replace("{URL_FROM_PEXELS_1}", image_urls[0])
        temp_html_content = temp_html_content.replace("{URL_FROM_PEXELS_2}", image_urls[1])

        for idx in range(3):  # This will loop over indices 0, 1, and 2
            temp_html_content = temp_html_content.replace(f"SLIDE_TITLE_{idx + 1}", slides_content[idx]["title"])
            temp_html_content = temp_html_content.replace(f"SLIDE_DESC_{idx + 1}", slides_content[idx]["description"])

        with open(temp_html_filename, 'w') as file:
            file.write(temp_html_content)
    else:
        print("Not enough images or content returned.")
        exit()

    async def screenshot(url, delay, num_screenshots, interval):
        browser = await launch(args=['--no-sandbox'])
        page = await browser.newPage()
        screenshot_paths = []
        extend_screenshot_time = int(2 / interval)

        try:
            await page.goto(url, waitUntil=["domcontentloaded", "load"], timeout=120000)
            await asyncio.sleep(delay)

            for _ in range(num_screenshots):
                # Check if the animation is complete
                is_complete = await page.evaluate('''() => {
                    return document.querySelector('.fade-in-text.completed') !== null;
                }''')

                if is_complete:
                    break

                screenshot_path = f"screenshot_{unique_id}_{len(screenshot_paths)}.jpg"
                screenshot_paths.append(screenshot_path)
                await page.screenshot({'path': screenshot_path, 'type': 'jpeg', 'quality': 50})
                await asyncio.sleep(interval)
        except Exception as e:
            print("Error occurred:", e)
        finally:
            await page.close()
            await browser.close()

        return screenshot_paths

    def generate_audio_from_scripts(scripts, voice="Bella", model="eleven_monolingual_v1"):
        audio_files = []
        for script in scripts:
            audio = generate(text=script, voice=voice, model=model)
            audio_filename = f"audio_{unique_id}_{len(audio_files)}.mp3"
            with open(audio_filename, "wb") as f:
                f.write(audio)
            audio_files.append(audio_filename)
        return audio_files

    def images_to_video(screenshot_paths, video_name, fps):
        if not screenshot_paths:
            print("No screenshots available to create a video.")
            return

        duration = 1.0 / fps
        clips = [ImageClip(m).set_duration(duration) for m in screenshot_paths]

        try:
            concat_clip = concatenate_videoclips(clips, method="compose")
            concat_clip.write_videofile(video_name, fps=fps)
        except Exception as e:
            print("Error while creating video:", e)

    scripts = [slide['script'] for slide in slides_content]
    audio_files = generate_audio_from_scripts(scripts)

    screenshot_interval = 0.03
    fps = 1.0 / screenshot_interval
    num_screenshots = int(20 / screenshot_interval)
    file_url = f"file://{os.path.abspath(temp_html_filename)}"
    screenshots = await screenshot(file_url, 1, num_screenshots, screenshot_interval)

    images_to_video(screenshots, 'presentation_15sec_{unique_id}.mp4', fps)

    video_clip = VideoFileClip('presentation_15sec_{unique_id}.mp4')
    # Combine all audio files
    audio_clips = [AudioFileClip(audio) for audio in audio_files]
    combined_audio = concatenate_audioclips(audio_clips)

    # Set the concatenated audio to the video
    video_with_audio = video_clip.set_audio(combined_audio)
    video_with_audio.write_videofile(final_video_filename  , codec="libx264", audio_codec="aac")

    # After creating the video, upload it to Azure Blob Storage
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=final_video_filename)

    with open(final_video_filename, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)



    ad_data={
       "url": blob_client.url,  # or provide a full URL if needed
       "date": datetime.now(),
       "creator": "some_creator_name"  # you can replace this with dynamic data
    }

    result = ads_collection.insert_one(ad_data)

    # Cleanup local files
    os.remove( final_video_filename)
    for audio in audio_files:
        os.remove(audio)
    for screenshot in screenshots:
        os.remove(screenshot)
    os.remove(temp_html_filename)
    print("video_url", blob_client.url)
    # Return the URL to the uploaded video
    return {"video_url": blob_client.url}


@app.post("/createpresentation")
async def create_ad_video(prompt: str = Body(...)):
    unique_id = uuid4().hex

    def create_audio_files(html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        slides = soup.find_all(class_='slide')

        audios = []
        durations = []
        for i, slide in enumerate(slides):
            text = slide.get_text()
            audio = generate(
                text=text,
                voice="Bella",
                model="eleven_monolingual_v1"
            )

            audio_path = f"audio_{i}.wav"
            with open(audio_path, 'wb') as f:
                f.write(audio)

            # Get audio duration
            data, samplerate = sf.read(audio_path)
            duration = len(data) / samplerate
            durations.append(duration)

            audios.append(audio_path)

        return audios, durations, len(slides)

    async def create_chat_completion():
        chat_completion_resp = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You have the capability to generate elegant presentations along with their content using only HTML. Additionally, you also author the content for the presentations. It is essential to structure the HTML content in 'slide' divisions, each enclosed in a <div> with class 'slide'. This allows us to create corresponding audio blocks for each slide. Here's an example: <div class=\"slide\"> <!-- Slide 1 content goes here --> </div> This helps us manage and synthesize audio content on a slide-by-slide basis. Please ensure this structure for optimal results."
                 },
                {"role": "user",
                 "content": prompt
                 }
            ]
        )
        return chat_completion_resp.choices[0].message['content']

    async def save_html(html_content, file_name):
        async with aiofiles.open(file_name, 'w') as f:
            await f.write(html_content)

    async def screenshot(url, delay, num_screenshots):
        browser = await launch()
        pages = await browser.pages()
        page = pages[0] if pages else await browser.newPage()

        try:
            await page.goto(url, timeout=60_000)
        except pyppeteer.errors.TimeoutError as e:
            print(f"Timed out with error: {e}")
            await browser.close()
            return []

        await asyncio.sleep(delay)

        screenshot_paths = []
        for i in range(num_screenshots):
            screenshot_path = f"screenshot_{unique_id}_{i}.png"
            screenshot_paths.append(screenshot_path)
            await page.screenshot({'path': screenshot_path})
            await page.evaluate("window.scrollBy(0, window.innerHeight);")
            await asyncio.sleep(delay)

        await browser.close()
        return screenshot_paths

    def images_to_transition_video(screenshot_paths, audio_paths, video_name, fps, durations):
        if not screenshot_paths:
            print("No screenshots to process.")
            return

        if len(screenshot_paths) != len(audio_paths):
            print("The number of images and audio files do not match.")
            return

        clips = []
        for img_path, audio_path, duration in zip(screenshot_paths, audio_paths, durations):
            audio_clip = AudioFileClip(audio_path)
            print(f"Loaded audio clip with duration: {audio_clip.duration} seconds")
            print(f"Linked audio file {audio_path} to video file {img_path}")  # Now printing paths instead of data
            clip = ImageClip(img_path).set_duration(duration)
            final_clip = CompositeVideoClip([clip.set_audio(audio_clip)])
            clips.append(final_clip)

        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(video_name, fps=fps)

    delay = 5  # seconds
    video_name = f'presentation_{unique_id}.mp4'
    fps = 1.0
    html_file_name = f"generated_{unique_id}.html"

    html_content = await create_chat_completion()
    await save_html(html_content, html_file_name)

    audios, durations, num_screenshots = create_audio_files(html_content)

    file_url = f"file://{os.path.abspath(html_file_name)}"
    screenshots = await screenshot(file_url, delay, num_screenshots)

    images_to_transition_video(screenshots, audios, video_name, fps, durations)

    # After creating the video, upload it to Azure Blob Storage
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=video_name)

    with open(video_name, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    ad_data={
        "url": blob_client.url,  # or provide a full URL if needed
        "date": datetime.now(),
        "creator": "some_creator_name"  # you can replace this with dynamic data
    }

    result = presentations_collection.insert_one(ad_data)

    print("video_url" + blob_client.url)
    return {"video_url": blob_client.url}


@app.get("/get_ads")
async def get_ads():
    ads = list(ads_collection.find({}))
    for ad in ads:
        ad['_id'] = str(ad['_id'])
    return ads

@app.get("/get_presentations")
async def get_presentations():
    presentations = list(presentations_collection.find({}))
    for presentation in presentations:
        presentation['_id'] = str(presentation['_id'])
    return presentations



@app.websocket("/hello")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    name = await websocket.receive_text()
    greeting = f"Hi {name}"
    await websocket.send_text(greeting)


@app.get("/connect")
def connect():
    return {"message": "Connected"}

if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI app using uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000)
