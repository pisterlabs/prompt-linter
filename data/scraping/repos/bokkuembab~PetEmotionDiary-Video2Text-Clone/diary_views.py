import math
import os
import pickle
import time
from glob import glob

import autogluon
import cv2
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import torch
import torchvision.transforms as T
from autogluon.tabular import TabularDataset, TabularPredictor
from django.conf import settings
from PIL import Image

from ai.apps import AiConfig

#####################################################################################################


# 객체 탐지 함수
def detecting(input_data):
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(input_data["video"])

    # 프레임 수 확인
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # # 모델 로드 (구글 드라이브에 있는 yolo폴더 직접 가져옴)
    # model = torch.hub.load('/content/drive/MyDrive/빅프로젝트/data/yolov5', 'custom', path='yolov5s.pt', source='local')

    frame_interval = int(cap.get(cv2.CAP_PROP_FPS))  # 1초당 1프레임
    frame_number = 0

    det = []
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        if frame_number % frame_interval != 0:
            continue

        # 프레임을 이미지로 변환
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 객체 탐지
        results = AiConfig.yolo5s(image)

        # 현재 프레임의 탐지 결과
        objects = results.pandas().xyxy[0]

        # 일정 스코어 이상인 객체 선택
        threshold = 0.7  # 일정 스코어
        objects = objects[objects["confidence"] >= threshold]

        # 탐지된 객체
        for _, obj in objects.iterrows():
            class_name = obj["name"]

            det.append(class_name)
            confidence = obj["confidence"]

    # 비디오 캡처 객체 해제
    cap.release()

    det = list(set(det))

    if input_data["pet"].kind in det:
        det.remove(input_data["pet"].kind)

    return str(det)[1:-1]


#####################################################################################################


# 동영상 이미지 분할 함수
def extract_frames(video_path, output_dir, diary_id):
    num_frames = 5  # 동영상을 이미지로 나눌 숫자 (5개 이미지 분할)

    # 동영상 파일 열기
    video = cv2.VideoCapture(video_path)

    # 동영상 정보 가져오기
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # 이미지 추출을 위한 프레임 간격 계산
    frame_interval = total_frames // num_frames

    # 이미지 추출을 위한 변수 초기화
    frame_count = 0
    saved_count = 0

    while frame_count < total_frames:
        # 프레임 읽기
        ret, frame = video.read()

        if not ret:
            break

        # frame_interval 마다 이미지 저장
        if frame_count % frame_interval == 0:
            output_path = f"{output_dir}/{diary_id}frame_{saved_count}.jpg"
            if saved_count == 0:
                output_path = (
                    f"static/media/split_imgs/{diary_id}frame_{saved_count}.jpg"
                )

            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_count += 1

    # 동영상 파일 닫기
    video.release()


#####################################################################################################


# 이미지들의 keypoint 예측 함수
def pred_keypoints(frames, animal):
    columns = [
        "1_x",
        "1_y",
        "2_x",
        "2_y",
        "3_x",
        "3_y",
        "4_x",
        "4_y",
        "5_x",
        "5_y",
        "6_x",
        "6_y",
        "7_x",
        "7_y",
        "8_x",
        "8_y",
        "9_x",
        "9_y",
        "10_x",
        "10_y",
        "11_x",
        "11_y",
        "12_x",
        "12_y",
        "13_x",
        "13_y",
        "14_x",
        "14_y",
        "15_x",
        "15_y",
    ]

    if animal == "Dog":
        keypoint_model = AiConfig.dog_skeleton_model
    elif animal == "Cat":
        keypoint_model = AiConfig.cat_skeleton_model

    key_df = pd.DataFrame(columns=columns)
    count = 1

    for frame in frames:
        image = cv2.imread(frame, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = image.transpose(2, 0, 1)
        image = [torch.as_tensor(image, dtype=torch.float32)]

        pred = keypoint_model(image)
        keypoints = pred[0]["keypoints"].detach().numpy().copy()[0]
        image = cv2.imread(frame, cv2.COLOR_BGR2RGB)
        keypoints = keypoints[:, :2]

        key_df.loc[len(key_df)] = keypoints[:, :2].reshape(-1)

    return key_df


#####################################################################################################


# keypoint들의 시퀀스 데이터 추출 함수
def extract_features(df, animal):
    seq = df.values
    max_seq_legth = 0

    if animal == "Dog":
        max_seq_length = 14
    elif animal == "Cat":
        max_seq_length = 180

    sequence = np.pad(seq, ((0, max_seq_length - len(seq)), (0, 0)), "constant")

    features = []

    # 관절 각도 추출
    for i in range(len(sequence) - 2):
        joint_angles = []
        for joint_idx in range(1, 16):
            p1 = sequence[
                i, (joint_idx - 1) * 2 : joint_idx * 2
            ]  # 관절 좌표 (x, y) 시퀀스의 현재 프레임
            p2 = sequence[
                i + 1, (joint_idx - 1) * 2 : joint_idx * 2
            ]  # 관절 좌표 (x, y) 시퀀스의 다음 프레임
            p3 = sequence[
                i + 2, (joint_idx - 1) * 2 : joint_idx * 2
            ]  # 관절 좌표 (x, y) 시퀀스의 다다음 프레임

            vector1 = p1 - p2
            vector2 = p3 - p2

            # 관절 각도 계산 (라디안 단위)
            angle = math.atan2(vector2[1], vector2[0]) - math.atan2(
                vector1[1], vector1[0]
            )
            joint_angles.append(angle)

        features.extend(joint_angles)

    # 속도 추출
    # 수정된 코드
    sequence = sequence.astype(float)  # sequence를 실수형으로 변환
    velocities = np.linalg.norm(np.diff(sequence, axis=0), axis=1)
    features.extend(velocities)

    # 가속도 추출
    accelerations = np.linalg.norm(np.diff(sequence, n=2, axis=0), axis=1)
    features.extend(accelerations)

    # 방향 변화 추출
    direction_changes = np.diff(np.arctan2(sequence[:, 1::2], sequence[:, ::2]), axis=0)
    features.extend(direction_changes.flatten())

    if animal == "Cat":
        features = features[:400]

    return features


#####################################################################################################


# 반려동물 감정 예측 함수
def predict_emotion(input_df, animal):
    if animal == "Dog":
        emotion_predictor = AiConfig.dog_emotion_predictor
    elif animal == "Cat":
        emotion_predictor = AiConfig.cat_emotion_predictor

    pred = emotion_predictor.predict(input_df)

    return pred[0]


#####################################################################################################
# 반려동물 행동 예측 함수
def predict_action(input_df, animal):
    if animal == "Dog":
        action_predictor = AiConfig.dog_action_predictor
    elif animal == "Cat":
        action_predictor = AiConfig.cat_action_predictor

    pred = action_predictor.predict(input_df)

    return pred[0]


#####################################################################################################


# chatGPT API 일기 요청 함수
def chatGPT(input_data, emotion, action, detected_objs):
    openai.api_key = settings.OPEN_API_KEY
    activity = {"A": "모험", "L": "안주"}
    relationship = {"E": "외향", "I": "내향"}
    proto_dog = {"C": "교감", "W": "본능"}
    dependence = {"T": "신뢰", "N": "필요"}

    prompt = ""

    if input_data["pet"].kind == "Dog":
        prompt = f"""강아지 이름: {input_data['pet'].name}
강아지 성별 : {input_data['pet'].gender}
강아지 감정: {emotion}
강아지 행동: {action}
강아지 성격: {activity[input_data['personality'].activity]}, {relationship[input_data['personality'].relationship]}, {proto_dog[input_data['personality'].proto_dog]}, {dependence[input_data['personality'].dependence]}
강아지가 보이는 것: {detected_objs}

주인 이름: {input_data['pet'].owner_name}

쓰고 싶은 내용: {str(input_data['add_content'])[1:-1]}

참고사항: 주인 - 주인 이름, 나 - 강아지 이름

위 사항을 기반해 강아지가 쓴 일기처럼 강아지 시점에서 일기를 써주세요.

일기를 토대로 제목도 작성해주세요 
일기를 토대로 키워드도 작성해주세요

일기 형식 : "title: 편안한 하루

diary_content: 오늘은 정말 편안한 하루였어멍. 집사가 나에게 소중한 이름 냐옹이라고 부르면서 마주쳐줬어멍.

keywords: 편안, 안정, 누워있기" 

위 모든 결과 분량 제한: 900자
    """

    elif input_data["pet"].kind == "Cat":
        prompt = f"""고양이 이름: {input_data['pet'].name}
고양이 성별 : {input_data['pet'].gender}
고양이 감정: {emotion}
고양이 행동: {action}
고양이 성격: {activity[input_data['personality'].activity]}, {relationship[input_data['personality'].relationship]}, {proto_dog[input_data['personality'].proto_dog]}, {dependence[input_data['personality'].dependence]}
고양이가 보이는 것: {detected_objs}

주인 이름: {input_data['pet'].owner_name}

쓰고 싶은 내용: {str(input_data['add_content'])[1:-1]}

참고사항: 주인 - 주인 이름, 나 - 고양이 이름

위 사항을 기반해 고양이가 쓴 일기처럼 고양이 시점에서 일기를 써주세요.

일기를 토대로 제목도 작성해주세요 
일기를 토대로 키워드도 작성해주세요

일기 형식 : "title: 편안한 하루

diary_content: 오늘은 정말 편안한 하루였어냥. 집사가 나에게 소중한 이름 냐옹이라고 부르면서 마주쳐줬어냥.

keywords: 편안, 안정, 누워있기"

위 모든 결과 분량 제한: 900자
    """

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
    )
    result = completion.choices[0].message.content

    return result


#####################################################################################################
def parse_result(result_string):
    result = {}

    # Split by new lines
    lines = result_string.split("\n")

    for line in lines:
        # Split by colon
        key_value = line.split(": ")

        if len(key_value) == 2:
            key = key_value[0]
            value = key_value[1]

            # If the key is "키워드", split the value by comma
            if key == "keywords":
                value = [v.strip() for v in value.split(",")]

            result[key] = value

    return result


# 일기 쓰기 함수
def create_diary(context):
    images_dir_path = settings.MEDIA_URL + "split_imgs"
    diary_id = str(context["diary_id"])

    # 동영상에서 분할한 이미지들을 저장할 폴더 (추가 필요)

    detected_objs = detecting(context)  # 객체 탐지 추출

    extract_frames(context["video"], images_dir_path, diary_id)  # 동영상에서 이미지로 분할후 폴더에 저장

    frames = glob(images_dir_path + "*.jpg")  # 이미지들의 경로를 저장한 리스트
    frames.sort()  # 시간순으로 정렬

    key_df = pred_keypoints(
        frames, context["pet"].kind
    )  # 이미지들의 keypoint 예측값들을 저장한 데이터프레임

    input_data = extract_features(key_df, context["pet"].kind)  # keypoint들을 시퀀스 데이터로 추출

    AiConfig.input_df_col_name.loc[0] = input_data

    input_df = TabularDataset(AiConfig.input_df_col_name)  # 감정 / 행동 모델에 넣을 input_df

    pet_emotion = predict_emotion(input_df, context["pet"].kind)  # 반려동물 감정 예측
    pet_action = predict_action(input_df, context["pet"].kind)  # 반려동물 감정 예측

    result = chatGPT(
        context, pet_emotion, pet_action, detected_objs
    )  # chatGPI 일기 쓰기 요청

    result_dict = parse_result(result)

    return result_dict
