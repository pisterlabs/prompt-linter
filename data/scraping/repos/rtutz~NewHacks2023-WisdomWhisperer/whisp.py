# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
from flask import Blueprint, jsonify, request, make_response
import openai
import os
import shutil
import dotenv
import re
import redis
import vertexai
from vertexai.language_models import TextGenerationModel
from yt_dlp import YoutubeDL

from .documents import upload
from .courses import getCourses, addCourse

dotenv.load_dotenv()

whisp = Blueprint('whisp', __name__)

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import (
    OpenAIWhisperParser,
)
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY
PROCESSED_DB_HOST = os.getenv('PROCESSED_DB_HOST')
PROCESSED_DB_PORT = os.getenv('PROCESSED_DB_PORT')
PROCESSED_DB_PASSWORD = os.getenv('PROCESSED_DB_PASSWORD')
VIDEO_DB_HOST = os.getenv('VIDEO_DB_HOST')
VIDEO_DB_PORT = os.getenv('VIDEO_DB_PORT')
VIDEO_DB_PASSWORD = os.getenv('VIDEO_DB_PASSWORD')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./GACKey.json"
PROJECT_ID = os.getenv('PROJECT_ID')


def transcribe(yturl):
    return_text = ""
    try:
        urls = [yturl]
        regex = "^((?:https?:)?\/\/)?((?:www|m)\.)?((?:youtube(-nocookie)?\.com|youtu.be))(\/(?:[\w\-]+\?v=|embed\/|live\/|v\/)?)([\w\-]+)(\S+)?$"
        uuid = re.search(regex, yturl, re.IGNORECASE).group(6)
        try:
            # Directory to save audio files
            save_dir = "../Downloads/YouTube"
            # https://stackoverflow.com/questions/19377262/regex-for-youtube-url

            # Transcribe the videos to text
            loader = GenericLoader(YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParser())
            docs = loader.load()
            try:
                for video in docs:
                    print(video.page_content)
                    return_text += " " + video.page_content
                for filename in os.listdir(save_dir):
                    file_path = os.path.join(save_dir, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print('Failed to delete %s. Reason: %s' % (file_path, e))
                response = {"transcription": str(return_text), "uuid": uuid}
                return response
            except Exception as e:
                print(e)
                raise Exception("We hit an error")
        except Exception as e:
            print(e)
            raise Exception("Error transcribing")
    except Exception as e:
        print(e)
        raise Exception("Error with URL")


@whisp.route('/add', methods=['POST'])
def add():
    try:
        data = request.get_json()
        yturl = data.get('yturl')
        course = data.get('course')
        regex = "^((?:https?:)?\/\/)?((?:www|m)\.)?((?:youtube(-nocookie)?\.com|youtu.be))(\/(?:[\w\-]+\?v=|embed\/|live\/|v\/)?)([\w\-]+)(\S+)?$"
        uuid = re.search(regex, yturl, re.IGNORECASE).group(6)
        redis_client = redis.Redis(
            host=PROCESSED_DB_HOST,
            port=PROCESSED_DB_PORT,
            password=PROCESSED_DB_PASSWORD)
        # https://redis-py.readthedocs.io/en/stable/commands.html#redis.commands.core.CoreCommands.sismember
        if not redis_client.sismember(course, uuid) == 0:
            redis_client.close()
            return jsonify("Video has already been processed"), 400
        else:
            try:
                redis_client.close()
                transcript = transcribe(yturl)
                transcript = transcript['transcription']

                video_title = getTitle(yturl)
                summary = text_summarization(0.1, transcript)
                try:
                    redis_client = redis.Redis(
                        host=VIDEO_DB_HOST,
                        port=VIDEO_DB_PORT,
                        password=VIDEO_DB_PASSWORD)

                    redis_client.hset(
                        uuid,
                        mapping={
                            "title": video_title,
                            "transcript": transcript,
                            "summary": summary
                        },
                    )
                    redis_client.close()
                    redis_client = redis.Redis(
                        host=PROCESSED_DB_HOST,
                        port=PROCESSED_DB_PORT,
                        password=PROCESSED_DB_PASSWORD)
                    redis_client.sadd(course, uuid)
                    redis_client.close()
                    print(upload(transcript, course))
                except:
                    return jsonify("We hit an error writing to database"), 500

                return jsonify({
                    "uuid": uuid,
                    "course": course,
                    "title": video_title,
                    "transcript": transcript,
                    "summary": summary
                }), 200
            # except:
            #     return jsonify("We hit an error transcribing"), 500
            except Exception as e:
                return jsonify("We hit an error transcribing", e), 500
    # except:
    #     return jsonify("Error with URL"), 400
    except Exception as e:
        return jsonify("Error with URL", e), 400

# https://cloud.google.com/vertex-ai/docs/samples/aiplatform-sdk-summarization
def text_summarization(
        temperature: float,
        transcription: str
) -> str:
    """Summarization Example with a Large Language Model"""

    vertexai.init(project=PROJECT_ID)
    # TODO developer - override these parameters as needed:
    parameters = {
        "temperature": temperature,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
        "top_p": 0.95,
        # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
    }

    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = model.predict(
        """Provide a summary with about 250 words for the following: """ + transcription,
        **parameters,
    )
    print(f"Response from Model: {response.text}")

    return response.text

def getTitle(yturl):
    with YoutubeDL() as ydl:
        info_dict = ydl.extract_info(yturl, download=False)
        video_title = info_dict.get('title', None)
        return video_title

@whisp.route('/notes', methods=['POST'])
def notes():
    return_array = []
    data = request.get_json()
    course = data.get('course')
    redis_client = redis.Redis(
        host=PROCESSED_DB_HOST,
        port=PROCESSED_DB_PORT,
        password=PROCESSED_DB_PASSWORD)
    videos = redis_client.smembers(course)
    redis_client.close()
    redis_client = redis.Redis(
        host=VIDEO_DB_HOST,
        port=VIDEO_DB_PORT,
        password=VIDEO_DB_PASSWORD)
    i = 0
    for item in videos:
        if len(str(item, encoding='utf-8')) > 0:
            video_info = redis_client.hgetall(item)
            # print(video_info)
            return_array.append({str(item, encoding='utf-8'): {"title": str(video_info.get(b'title'), encoding='utf-8'), "summary": str(video_info.get(b'summary'), encoding='utf-8')}})
            print({str(item, encoding='utf-8'): {"title": str(video_info.get(b'title'), encoding='utf-8')}})
            i += 1
    redis_client.close()
    return jsonify({"notes": return_array, "course": course})

@whisp.route('/getCourseList', methods=['POST'])
def getCourseList():
    return_array = getCourses()
    return jsonify({"courses": return_array}), 200

@whisp.route('/addCourseList', methods=['POST'])
def addCourseList():
    data = request.get_json()
    course_name = data.get('course')
    if addCourse(course_name):
        return jsonify({"message": "Success"}), 200
    else:
        return jsonify({"message": "Not added, exists or wrong"}), 400
