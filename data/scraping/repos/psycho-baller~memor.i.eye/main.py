import cohere
from datetime import datetime
from flask import Flask, request
from annoy import AnnoyIndex

from shared import get_conn, get_cohere_client

import subprocess

import os

BASE_URL = "http://20.25.130.61"

co = get_cohere_client()

app = Flask(__name__)

@app.route("/upload", methods = ["POST"])
def upload_image():
    data = request.get_json()
    with get_conn() as conn:
        with conn.cursor() as cur:
            url, timestamp, latitude, longitude = data["url"], data["time"], data["latitude"], data["longitude"]
            cur.execute("""
                INSERT INTO Image (url, time, latitude, longitude) VALUES (%s, %s, %s, %s) RETURNING id
            """, (url, datetime.fromtimestamp(timestamp), latitude, longitude))
            newid = cur.fetchone()
            subprocess.Popen(['python3', './upload.py', str(newid['id'])])
    return ""

@app.route("/test_photo", methods = ["POST"])
def test_photo():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT latitude, longitude from Location ORDER BY time DESC LIMIT 1")
            row = cur.fetchone()
            latitude = row['latitude']
            longitude = row['longitude']
            gaze_x = request.args['gaze_x']
            gaze_y = request.args['gaze_y']
            time = datetime.fromtimestamp(float(request.args['time']))
            cur.execute("""INSERT INTO Image (latitude, longitude, gaze_x, gaze_y, time) VALUES (%s, %s, %s, %s, %s) RETURNING id""", (latitude, longitude, gaze_x, gaze_y, time))
            _id = str(cur.fetchone()['id'])
            path = 'pictures/' + _id + '.jpg'
            with open(path, 'wb') as f:
                f.write(request.data)
            subprocess.Popen(['python3', './face_find.py', str(_id)])
    return ""



@app.route("/images")
def images():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id,time,latitude,longitude,note FROM Image WHERE valid = true")
            result = []
            for image in cur:
                image['url'] = BASE_URL + '/pictures/' + str(image['id']) + '.jpg'
                result.append(image)
            return result


@app.route("/search")
def search():
    query = request.args['q']
    search_index = AnnoyIndex(4096, "angular")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, embed FROM Image WHERE embed IS NOT NULL")
            embeds = [image for image in cur]
            for i, embed in enumerate(embeds):
                search_index.add_item(i, [float(x) for x in embed["embed"].split(",")])
            search_index.build(15)
            query_embed = co.embed(texts=[query]).embeddings[0]
            similar_item_ids = search_index.get_nns_by_vector(query_embed, 10, include_distances=True)
            print(similar_item_ids)
            return [embeds[item_id]["id"] for item_id in similar_item_ids[0]]

@app.route("/note/<int:image_id>", methods = ["POST"])
def upload_note(image_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE Image SET note = %s WHERE id = %s", 
                    (request.data.decode(), image_id))
    return ""

@app.route("/latest_loc")
def latest_loc():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT latitude, longitude, altitude FROM Location ORDER BY time DESC LIMIT 1")
            return cur.fetchone()
