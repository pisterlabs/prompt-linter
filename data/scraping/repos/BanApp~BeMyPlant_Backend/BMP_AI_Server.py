import jwt
import pymysql
from pymysql import cursors
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from dbutils.pooled_db import PooledDB
from fastapi import FastAPI, Request, HTTPException, Header, Depends
from pydantic import BaseModel
import requests
import openai
import json
from pymongo import MongoClient
from sklearn.ensemble import IsolationForest

# MariaDB 연결 풀 설정
db_pool = PooledDB(
    creator=pymysql,
    maxconnections=10,  # 최대 연결 수 설정
    host="",
    port=,
    user="",
    password="",
    database="",
    cursorclass=cursors.DictCursor
)

# MongoDB 연결 정보
MONGO_URI = ""  # MongoDB 서버 주소에 맞게 수정하세요
DB_NAME = ""  # 데이터베이스 이름에 맞게 수정하세요

# MongoDB 연결 클라이언트
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]

app = FastAPI()

# Set your OpenAI API key here
openai.api_key = ''

class PlantImageRequest(BaseModel):
    color: str
    species: str
    pot_color: str

@app.post("/api/generate_plant_images")
async def generate_plant_images(request: Request, plant_image_request: PlantImageRequest):
    color = plant_image_request.color
    species = plant_image_request.species
    pot_color = plant_image_request.pot_color

    try:
        # Generate images using OpenAI API
        prompt = f"a cute indoor plant with green leaves on a letterless and a transparent background. {color} {species} and {pot_color} pots. In pixel art style"

        plant_response = openai.Image.create(prompt=prompt, n=10, size="512x512")

        # Get image URLs
        plant_image_urls = []
        for i in range(10):
            image_url = plant_response['data'][i]['url']
            plant_image_urls.append(image_url)

        return {"plant_image_urls": plant_image_urls}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class UserImageRequest(BaseModel):
    gender: str
    characteristic: str

@app.post("/api/generate_user_images")
async def generate_user_images(request: Request, user_image_request: UserImageRequest):
    gender = user_image_request.gender
    characteristic = user_image_request.characteristic

    try:
        # Generate images using OpenAI API
        prompt = f"Draw the face of {gender} with {characteristic} on a letterless and a transparent background in pixel art style."

        user_response = openai.Image.create(prompt=prompt, n=4, size="512x512")

        # Get image URLs
        user_image_urls = []

        for i in range(4):
            image_url = user_response['data'][i]['url']
            user_image_urls.append(image_url)

        return {"user_image_urls": user_image_urls}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Sentence Transformer 모델 호출
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# DataFrame 불러오기
df = pd.read_csv('final_dataset.csv')

# 문자열 형태의 임베딩 값을 파싱하고 NumPy 배열로 변환하는 함수
def parse_embedding(embedding_str):
    # 문자열에서 '[', ']'를 제거하고 ','를 기준으로 분할하여 실수로 변환
    values = [float(x) for x in embedding_str[1:-1].split(', ')]
    return np.array(values)

# 'embedding' 열의 값을 파싱한 NumPy 배열로 변경
df['embedding'] = df['embedding'].map(parse_embedding)

# JWT 토큰 설정
SECRET_KEY = ''
ALGORITHM = ''

# JWT 토큰 검증 함수
def verify_token(authorization: str = Header(...)):
    try:
        token = authorization.replace("Bearer ", "")
        payload = jwt.decode(token, options={"verify_signature": False})
        username = payload.get("sub")

        # 연결 풀을 사용하여 데이터베이스 연결을 가져오기
        connection = db_pool.connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM userinfo WHERE username=%s", (username,))
            user_info = cursor.fetchone()
        connection.close()

        if user_info is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_info
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/api/chatbot")
async def chat_bot(input: dict, user_info: dict = Depends(verify_token)):
    try:
        # 챗봇 실행
        text = input.get("input_text")
        ebd = model.encode(text)
        distances = df['embedding'].map(lambda x: cosine_similarity([ebd], [x]).squeeze())
        max_idx = np.argmax(distances)
        answer = df.iloc[max_idx]['A']
        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

class WeatherRequest(BaseModel):
    city: str

# MongoDB에서 최신 1441개의 데이터를 가져오는 함수
def get_latest_sensor_data(user_id: str):
    collection_name = user_id
    sensor_collection = db[collection_name]
    data_count = sensor_collection.count_documents({})

    if data_count >= 1441:
        # 데이터가 1441개 이상인 경우, 최신 1441개의 데이터를 가져옵니다.
        data = list(sensor_collection.find({}, projection={"airTemp": 1, "airHumid": 1, "soilHumid": 1, "lightIntensity": 1}).sort('_id', -1).limit(1441))
    else:
        # 데이터가 1441개 미만인 경우, 모든 데이터를 가져옵니다.
        data = list(sensor_collection.find({}, projection={"airTemp": 1, "airHumid": 1, "soilHumid": 1, "lightIntensity": 1}))
    return data

@app.post("/api/weather_and_status")
async def weather_and_status(weather_request: WeatherRequest, user_info: dict = Depends(verify_token)):
    city = weather_request.city
    weather_apiKey = ''
    lang = 'kr'  # 언어
    units = 'metric'  # 화씨 온도를 섭씨 온도로 변경
    user_id = user_info['username']
    sensor_data = get_latest_sensor_data(user_id)

    try:
        if sensor_data:
            # 최신 데이터 하나만 가져오기
            latest_data = sensor_data[0]

            latest_data = np.array([
                latest_data["airTemp"],
                latest_data["airHumid"],
                latest_data["soilHumid"],
                latest_data["lightIntensity"]
            ])

            latest_data = latest_data.reshape(1, -1)

            training_data = sensor_data[1:]

            training_data = np.array([
                [data["airTemp"], data["airHumid"], data["soilHumid"], data["lightIntensity"]]
                for data in training_data
            ])

            model = IsolationForest(contamination=0.01)  # 이상치 비율에 따라 조절
            model.fit(training_data)
            y_pred = model.predict(latest_data)
            if y_pred[0] == -1:
                # 이상치로 판별
                stat = 0
            else:
                # 정상 데이터
                stat = 1

            api = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_apiKey}&lang={lang}&units={units}"
            result = requests.get(api)
            result = json.loads(result.text)
            weather_result = {"name": result['name'],
                             "weather": result['weather'][0]['main'],
                             "temperature": result['main']['temp'],
                             "humidity" : result['main']['humidity'],
                             "status" : stat}
            return weather_result
        else:
            raise HTTPException(status_code=404, detail="센서 데이터를 찾을 수 없습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

