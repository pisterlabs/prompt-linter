from twitchstream.outputvideo import TwitchBufferedOutputStream
from date_helper import convert_to_human_readable
from flask_sock import Sock as WebSocket
from deso_helper import upload_to_deso
from cohere_helper import classify
from db_helper import db, db_raw
from fix_cors import fix_cors
from base64 import b64decode
from PIL import Image
from flask import *
import datetime
import numpy
import uuid
import io

app = Flask(__name__)
app.config["SOCK_SERVER_OPTIONS"] = {"ping_interval": 25}
websocket = WebSocket(app)

def gen_uuid():
  return str(uuid.uuid4())

@websocket.route("/")
def mainws(ws):
  token = False
  image = False
  videostream = None
  while True:
    data = ws.receive()
    if data == "close":
      break
    if not token:
      token = data
      continue
    if not image:
      img = Image.open(io.BytesIO(b64decode(data)))
      videostream = TwitchBufferedOutputStream(twitch_stream_key=token, width=img.size[0], height=img.size[1], fps=30.0, ffmpeg_binary="/home/runner/Streaming-For-All/ffmpeg")
      image = True
    videostream.send_video_frame(numpy.array(Image.open(io.BytesIO(b64decode(data))))[...,:3])

@app.route("/__db__", methods=["GET"])
@fix_cors
def retdb():
  return jsonify(db_raw.dict())

@app.route("/event/all", methods=["GET"])
@fix_cors
def all_events():
  return jsonify(list(db_raw["events"].values()))

@app.route("/event/get/<eid>", methods=["GET"])
@fix_cors
def get_event(eid):
  return jsonify(db_raw["events"][eid])

@app.route("/event/create", methods=["POST"])
@fix_cors
def create_event():
  stuff = request.form.to_dict()
  eid = gen_uuid()
  stuff["time"] = convert_to_human_readable(stuff["date"])+" @ "+datetime.datetime.strptime(stuff["time"], "%H:%M").strftime("%I:%M %p").lower()
  del stuff["date"]
  stuff["eid"] = eid
  stuff["host"] = db_raw["users"][stuff["uid"]]["name"]
  stuff["contact"] = db_raw["users"][stuff["uid"]]["email"]
  stuff["profile"] = db_raw["users"][stuff["uid"]]["profile"]
  stuff["cover"] = upload_to_deso(request.files["cover"])
  pred = classify(stuff["description"])
  if len(pred) > 0:
    return pred
  else:
    db["events"][eid] = stuff
    return ""

@app.route("/user/create", methods=["POST"])
@fix_cors
def create_user():
  db["users"][request.json["uid"]] = request.json
  return ""

@app.route("/user/get/<uid>", methods=["GET"])
@fix_cors
def get_user(uid):
  return jsonify(db_raw["users"][uid])

app.run(host="0.0.0.0")
