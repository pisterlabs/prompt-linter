import lyricsgenius
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from multiprocessing import Pool
import json
import openai
import requests
import random
import os
from dotenv import load_dotenv
from PIL import Image
import io
import base64
import json

import psycopg

import cohere

# Load the environment variables from .env file
load_dotenv()

# Access the environment variables
oak = os.getenv("OPENAI_API_KEY")
gak = os.getenv("GENIUS_API_KEY")
pg_conn_string = os.getenv("DATABASE_URL")
cohere_key = os.getenv("COHERE_API_KEY")

co = cohere.Client(cohere_key)

conn = psycopg.connect(pg_conn_string, sslrootcert="./.postgresql/root.crt")
try:
    make_table = "CREATE TABLE Users (user_id VARCHAR, playlist_id VARCHAR, time TIMESTAMP, image_id VARCHAR, selected BOOL)"
    res =  conn.execute(make_table)
    conn.commit()
except:
    conn.rollback()

genius = lyricsgenius.Genius(gak)
openai.api_key = oak

def get_lyrics(title, artist):
    try:
        # Search for the song lyrics
        song = genius.search_song(title, artist)
        
        if song:
            return song.lyrics
        else:
            return NotImplemented
    
    except Exception as e:
        return None

# can only use 10,000 tokens per minute

def process_song(song):
    l = get_lyrics(*song)

    if l is None or type(l) is not str:
        return

    # messages = [ {"role": "system", "content": "Your job is to take in song lyrics and write describing elements of the lyrics. Your response should describe the colors. The shapes and objects. It should explain key details. It should also be under 100 words."} ]
    lyric_message = l.replace("\n", " ") 
    # messages.append({"role": "user", "content": lyric_message},)

    # chat = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo", messages=messages, max_tokens=800
    # )

    # reply = chat.choices[0].message.content

    reply = co.summarize(text = lyric_message, additional_command="Your job is to take in song lyrics and write describing elements of the lyrics. Your response should describe the colors. The shapes and objects. It should explain key details. It should also be under 100 words.")
    reply=reply.summary
    print(reply)

    return reply

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your specific allowed origins
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

@app.post("/create")
async def create_item(request: Request, username: str, playlist: str, response: Response):
    songs = json.loads(await request.body())
    songs = [(song,artist) for song,artist in songs.items()]

    songs = songs[0:6]

    with Pool() as pool:
        replies = pool.map(process_song, songs)

    replies = [reply for reply in replies if reply is not None]
    
    messages = [ {"role": "system", "content": "Given descriptions separated by | characters write a short yet descriptive prompt to generate an image representative of the descriptions. The resulting prompt should be more general, it should make one cohesive image. The prompt should describe an art style that matches the theme of the descriptions. The maximum length of the response should be under 100 words.."} ]
    descriptions = "|".join(replies)
    messages.append({"role": "user", "content": descriptions},)
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages, max_tokens=1000
    )

    output = chat.choices[0].message.content + ". There should never be text in the resulting image."

    response = openai.Image.create(
        prompt=output,
        n=4,
        size="1024x1024"
    )
    
    image_urls = [response['data'][i]['url'] for i in range(len(response['data']))]

    letters='abcdefghijklmnopqrstuvwxyz'

    ret_ids = []

    for url in image_urls:
        name = ''.join(random.choice(letters) for i in range(40))
        r = requests.get(url, allow_redirects=True)
        image_data = r.content
        image_stream = io.BytesIO(image_data)
        image = Image.open(image_stream)
        image.save("images/" + name + ".jpg",  optimize=True, quality=10)
        ret_ids.append(name)

    # write to DB 
    insert_sql = """
    INSERT INTO Users (user_id, playlist_id, time, image_id, selected)
    VALUES (%s, %s, CURRENT_TIMESTAMP, %s, %s)
    """

    for ret_id in ret_ids:
        conn.execute(insert_sql, (username, playlist, ret_id, False))
        conn.commit()

    response_data = json.dumps(ret_ids)

    headers = {"Access-Control-Allow-Origin": "*"}
    return Response(content=response_data, headers=headers)

@app.get("/image/{image_id}")
def get_image(image_id):
    headers = {"Access-Control-Allow-Origin": "*"}
    return FileResponse("images/" + image_id + ".jpg", headers=headers)

@app.get("/image_blob/{image_id}")
def get_image(image_id):
    headers = {"Access-Control-Allow-Origin": "*"}
    with open("images/" + image_id + ".jpg", "rb") as file:
        file_bytes = file.read()
    base64_encoded = base64.b64encode(file_bytes).decode('utf-8')
    return PlainTextResponse(base64_encoded, headers=headers)

@app.get("/featured/")
def get_featured():
    headers = {"Access-Control-Allow-Origin": "*"}
     
    select_sql = """
    SELECT image_id FROM Users WHERE selected = True ORDER BY time DESC LIMIT 10
    """
    res = conn.execute(select_sql).fetchall()
    conn.commit()

    values= []
    for row in res:
        values.append(row[0])

    return Response(content=json.dumps(values), headers=headers)

@app.get("/user/{user_id}")
def get_user(user_id: str):
    headers = {"Access-Control-Allow-Origin": "*"}
    # get everything associated with user
    select_sql = """
    SELECT playlist_id, image_id FROM Users WHERE user_id = %s
    """
    res = conn.execute(select_sql, (user_id,)).fetchall()
    conn.commit()

    playlist_to_images = {}

    # Iterate through the rows and populate the dictionary
    for row in res:
        playlist_id, image_id = row
        if playlist_id not in playlist_to_images:
            playlist_to_images[playlist_id] = []
        playlist_to_images[playlist_id].append(image_id)

    return Response(content=json.dumps(playlist_to_images), headers=headers)

@app.put("/select/{image_id}")
def select_image(image_id: str):
    headers = {"Access-Control-Allow-Origin": "*"}
    update_sql = """
    UPDATE Users
    SET selected = True
    WHERE image_id = %s
    """
    conn.execute(update_sql, (image_id,))
    conn.commit()
    return Response(content="success", headers=headers)


@app.get("/download/{image_id}")
async def download_file(image_id: str):
    file_name = image_id + ".jpg"
    file_path = "images/" + file_name
    download_name = "cover.jpg"

    update_sql = """
    UPDATE Users
    SET selected = True
    WHERE image_id = %s
    """
    conn.execute(update_sql, (image_id,))
    conn.commit()
    
    # Use the FileResponse class to send the file as a response
    return FileResponse(file_path, headers={"Content-Disposition": f"attachment; filename={download_name}"})
