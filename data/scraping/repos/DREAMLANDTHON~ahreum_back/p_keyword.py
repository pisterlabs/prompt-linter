from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from googleapiclient.discovery import build
from isodate import parse_duration
from googleapiclient.errors import HttpError
import openai
from config import OPENAI_KEY
from config import YOUTUBE_API_KEY


def get_video_info(video_id):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    try:
        request = youtube.videos().list(
            part="snippet",
            id=video_id
        )
        response = request.execute()
        title = response['items'][0]['snippet']['title']
        description = response['items'][0]['snippet']['description']
        return title, description
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
        return None
    
def get_key_wards(video_id):
    title, des, thum = get_video_info(video_id)
    openai.api_key = OPENAI_KEY
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You can't print anything other than wards and commas."},
        {"role": "system", "content": "You need to extract no more than three words."},
        {"role": "system", "content": "You shouldn't use parentheses after words when you extract words."},
        {"role": "user", "content": f"{title}라는 영상 제목과 {des}라는 영상 설명에서 영상의 특성을 나타내는 단어를 추출해줘"},
    ],
    temperature = 0.5,
    frequency_penalty = 0.6
    )
    re = response['choices'][0]['message']['content']
    ward_list = re.split(',')
    keyward_list = []
    for ward in ward_list:
        keyward_dict = {}
        keyward_dict['keyword'] = ward
        keyward_list.append(keyward_dict)
    return keyward_list



class keyword(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument("video_id", type=str, required=True, help="User ID is required.")
            args = parser.parse_args()

            video_id = args["video_id"]
            
            re = get_key_wards(video_id)
            return re
        
        except Exception as e:
            return {'error': str(e)}
