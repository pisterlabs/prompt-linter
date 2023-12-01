import pika
import time
from pathlib import Path
from flask import Flask, flash, request, redirect, url_for
import uuid
import psycopg2
from openai_functions import respond


UPLOAD_DIRECTORY = Path("files")
UPLOAD_DIRECTORY.mkdir(exist_ok=True)
MAX_AUDIO_PARSE_TIME = 10
POLLING_INTERVAL = 0.3

app = Flask(__name__)

def get_connection():
    conn  =psycopg2.connect(user="postgres", password="secret",
                              host="localhost", port="5533",
                              database="voice")
    conn.autocommit = True
    return conn

@app.route('/greet', methods=['GET'])
def get_uuid():
    print("PostgreSQL server information")
    id = uuid.uuid4()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO client (client_id, ip) VALUES (%s, %s);", (str(id), request.remote_addr))
    cur.execute(
        f"SELECT * FROM client;")
    print(cur.fetchall())
    conn.commit()
    cur.close()
    conn.close()
    return {"id": id}


@app.route('/speak', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        id =  request.form['id']
        print("id:", id)
        if not len(id):
            return
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            file_id = uuid.uuid4()
            filename = UPLOAD_DIRECTORY / f"{file_id}.{file.filename.split('.')[-1]}"
            print(filename)
            file.save(filename)
            connection = pika.BlockingConnection()
            channel = connection.channel()
            channel.queue_declare(queue=str(file_id))
            channel.basic_publish(exchange='', routing_key="test",
                                  body=bytes(str(filename)+"|"+str(file_id), "utf-8"))

            text = ""
            for i in range(int(MAX_AUDIO_PARSE_TIME/POLLING_INTERVAL)+1):
                time.sleep(0.3)
                method_frame, header_frame, body = channel.basic_get(str(file_id))
                if method_frame:
                    text = bytes.decode(body)
                    break
            channel.queue_delete(queue=str(file_id))
            connection.close()
            if len(text):
                print("New text", text)
                conn = get_connection()
                cur = conn.cursor()
                cur.execute(
                    f"SELECT query, response FROM session WHERE client_id='{id}' ORDER BY created_time;")
                log = cur.fetchall()
                cur.close()

                response = respond(text, log)
                cur = conn.cursor()
                cur.execute(
                    f"INSERT INTO session (client_id, query, response) "
                    f"VALUES (%s, %s, %s)", (str(id), text, response))
                conn.commit()
                cur.close()
                conn.close()

                conn = get_connection()
                cur = conn.cursor()
                cur.execute(
                    f"SELECT query, response FROM session WHERE client_id='{id}' ORDER BY created_time;")
                print(cur.fetchall())
                cur.close()
                conn.close()
            else:
                return {"query": "", "response": ""}
            return {"query": text, "response": response }