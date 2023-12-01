import os
import argparse
import cv2
import numpy as np
import pyttsx3
import openai
import time
from threading import Thread
import importlib.util

def initialize_tts_engine():
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('voice', 'english_rp+f4')
    return tts_engine

tts = None
object_detected_times ={}
openai.api_key = "REDACTED KEY"

def get_object_description(object_name):
    response = openai.Completion.create(
        engine="text-ada-001", 
        prompt=f"Give a really short description of a {object_name}.", 
        max_tokens=50
    )
    description = response.choices[0].text.strip()
    return description

def play_audio(tts, object_name):
    print("Play Audio")
    object_description = get_object_description(object_name)
    time.sleep(1)
    tts.say(f"{object_name} Detected.")
    tts.runAndWait()
    tts.say(object_description)
    tts.runAndWait()

def threaded_play_audio(tts, object_name):
    audio_thread = Thread(target=play_audio, args=(tts, object_name,))
    audio_thread.start()

# Other class and function definitions...

if __name__ == "__main__":
    args = parser.parse_args()
    tts = initialize_tts_engine()
    # Other setup code...

    def draw_object_box_and_label(frame, boxes, classes, scores, i):
        ymin = int(max(1,(boxes[i][0] * imH)))
        xmin = int(max(1,(boxes[i][1] * imW)))
        ymax = int(min(imH,(boxes[i][2] * imH)))
        xmax = int(min(imW,(boxes[i][3] * imW)))

        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

        object_name = labels[int(classes[i])] 
        label = '%s: %d%%' % (object_name, int(scores[i]*100)) 
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) 
        label_ymin = max(ymin, labelSize[1] + 10) 
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) 
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    def handle_detection_score(object_name, scores, i):
        if scores[i] > 0.64:  
            current_time = time.monotonic()

            if object_name not in object_detected_times:
                object_detected_times[object_name] = current_time

            if current_time - object_detected_times[object_name] > 2:
                threaded_play_audio(tts, object_name)
                object_detected_times[object_name] = current_time
        else:
            object_detected_times.pop(object_name, None)

    while True:
        t1 = cv2.getTickCount()
        frame1 = videostream.read()
        # Frame processing code omitted for brevity...

        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                draw_object_box_and_label(frame, boxes, classes, scores, i)
                object_name = labels[int(classes[i])] 
                handle_detection_score(object_name, scores, i)

        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        cv2.imshow('Object detector', frame)
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    videostream.stop()
