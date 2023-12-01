import cv2
import math
import openai
import speech_recognition as sr
import pygame
from playsound import playsound
from gtts import gTTS
import json
import os
import argparse

openai.api_key = "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"

def seslendir(kelime):
    if kelime:
        tts = gTTS(text=kelime, lang='tr')
        tts.save('seslendirilen.mp3')
        playsound('seslendirilen.mp3')
        os.remove('seslendirilen.mp3')

def get_custom_response(question):
    with open('custom_responses.json', 'r') as f:
        custom_responses = json.load(f)
    return custom_responses.get(question)

def chat_with_receptionist(message):
    custom_response = get_custom_response(message)
    if custom_response is not None:
        response_text = custom_response
    else:
        full_prompt = "Kullanıcı: " + message + "\nSen bir otelin lobisinde görevlisisin. Oteldeki müşterilere samimi ve sıcak cevaplar ver. Verdigin cevaplar 30 kelimeyi asla geçmesin. Ve asla otel haricinde sorulan sorulara cevap verme. Ve asla hiçbir şekilde kod yazma. Sadece anlaşılır cevaplar ver:"

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=full_prompt,
            temperature=0.7,
            max_tokens=50
        )

        response_text = str(response['choices'][0]['text']).strip('\n\n')

        sentences = response_text.split('. ')
        last_sentence = sentences[-1]
        if not last_sentence.endswith('.'):
            response_text = '. '.join(sentences[:-1])

    seslendir(response_text)

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes

parser = argparse.ArgumentParser()
parser.add_argument('--image')

args = parser.parse_args()

def get_gender_title(cinsiyet, isim):
    if cinsiyet == "female":
        return isim + " Hanım" 
    else:
        return isim + " Bey" 

def listen_and_respond():
    r = sr.Recognizer()  

    pygame.mixer.init()
    pygame.mixer.music.load("DING.mp3")
    pygame.mixer.music.play()

    print("Listening...")

    with sr.Microphone() as source:
        audio = r.listen(source)

    try:
        prompt = r.recognize_google(audio, language="tr-TR", show_all=False)
        print("You asked:", prompt)

        # Cinsiyet tahmini yap
        seslendir("Önce isminizi öğrenebilir miyim?")
        with sr.Microphone() as source:
            audio = r.listen(source)

        isim = r.recognize_google(audio, language="tr-TR", show_all=False)

        # Cinsiyet tahmini yap
        cinsiyet = perform_gender_prediction()

        # Hitap eklemek için gerekli işlemi yap
        tam_isim = get_gender_title(cinsiyet, isim)

        # Chat işlemlerini yap
        chat_with_receptionist(tam_isim)

    except sr.UnknownValueError:
        response_text = "Anlayamadım"
        seslendir(response_text)
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service;  {0}".format(e))
def perform_gender_prediction():
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    genderList = ['Male', 'Female']

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    video = cv2.VideoCapture(0)  # Kameranın 0 numaralı aygıt olduğunu varsayalım

    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            break

        resultImg, faceBoxes = highlightFace(faceNet, frame)

        if not faceBoxes:
            print("No face detected")

        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1] - padding): min(faceBox[3] + padding, frame.shape[0] - 1),
                        max(0, faceBox[0] - padding): min(faceBox[2] + padding, frame.shape[1] - 1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')

            return gender

    video.release()
    cv2.destroyAllWindows()

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
padding = 30

def main():
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    video = cv2.VideoCapture(args.image if args.image is not None else 0)  # Kameranın 0 numaralı aygıt olduğunu varsayalım
            
    while cv2.waitKey(1) < 0:
        hasFrame, frame = video.read()
        if not hasFrame:
            cv2.waitKey()
            break
            
        resultImg, faceBoxes = highlightFace(faceNet, frame)

        if not faceBoxes:
            print("No face detected")

        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1] - padding): min(faceBox[3] + padding, frame.shape[0] - 1),
                        max(0, faceBox[0] - padding): min(faceBox[2] + padding, frame.shape[1] - 1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years')

            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Detecting age and gender", resultImg)

        listen_and_respond()
    
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

