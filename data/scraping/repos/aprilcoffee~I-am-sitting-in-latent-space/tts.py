
import cv2  # We're using OpenCV to read video
import base64
import time
import openai
import os
import requests
import config 


def textPrompt(img):
    
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                "These are frames of a video. Create a short voiceover script in the style of David Attenborough. Only include the narration.",
                {"image": img.tolist(), "resize": 768},
            ],
        },
    ]
    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "api_key": config.openai_api_key,
        "headers": {"Openai-Version": "2020-11-07"},
        "max_tokens": 500,
    }

    result = openai.ChatCompletion.create(**params)
    print(result.choices[0].message.content)

def returnCameraIndexes():
    # checks the first 10 indexes.
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr

print(returnCameraIndexes())

# Open a specific camera (change the index as needed)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow('frame', frame)
        textPrompt(frame)
        
        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # Release the capture object
    cv2.destroyAllWindows()





while(False):

    ret,frame = cap.read()

    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                "These are frames of a video. Create a short voiceover script in the style of David Attenborough. Only include the narration.",
                *map(lambda x: {"image": x, "resize": 768}, frame),
            ],
        },
    ]
    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "api_key": config.openai_api_key,
        "headers": {"Openai-Version": "2020-11-07"},
        "max_tokens": 500,
    }

    result = openai.ChatCompletion.create(**params)
    print(result.choices[0].message.content)


    response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {config.openai_api_key}",
        },
        json={
            "model": "tts-1",
            "input": result.choices[0].message.content,
            "voice": "onyx",
        },
    )

    audio = b""
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        audio += chunk
    Audio(audio)

    key = cv2.waitKey(1)
    if key==32:
        new = False
    if key==27 or key==ord('q'):
        break
    done = True
    cv2.imshow('frame',frame)

print('direct out')
cap.release()
cv2.destroyAllWindows()