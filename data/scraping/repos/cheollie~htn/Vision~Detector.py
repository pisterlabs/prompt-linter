import cv2
import numpy as np
import time
import pyttsx3
from threading import Thread
import cohere
import requests
from IPython.display import Audio

class SpeechThread(Thread):
    def __init__(self, engine: pyttsx3.Engine, text: str):
        self.engine = engine
        self.text = text
        super(SpeechThread, self).__init__()

    def run(self):
        self.engine.say(self.text)
        self.newVoiceRate = 280
        self.engine.setProperty('rate',self.newVoiceRate)
        self.engine.runAndWait()
        #pass


class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath 
        self.modelPath = modelPath
        self.classesPath = classesPath
        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.net.setInputSize(320, 320)
        self.net.setInputScale (1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB (True)
        self.object_timestamps = {}
        self.object_duration = {}
        self.readClasses()
        self.engine = pyttsx3.init()
        self.co = cohere.Client('wqRmhTjacxHnbOJtkV2fQX8MK3swlx80QuUX6h0F') # This is your trial API key
    def readClasses(self):
        with open(self.classesPath, 'r') as f: self.classesList = f.read().splitlines()
        self.classesList.insert(0, '__Background__')
        print(self.classesList)
    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)
        if (cap.isOpened()==False): 
            print("Error opening file...") 
            return
        last_time = time.time()
        (success, image) = cap.read()
        while success:
            classLabelIDs, confidences, bboxs = self.net.detect (image, confThreshold = 0.4)
            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1,-1)[0])
            confidences = list(map (float, confidences))
            bboxIdx = cv2.dnn. NMSBoxes (bboxs, confidences, score_threshold = 0.35, nms_threshold = 0.2)
            if len(bboxIdx) != 0:
                for i in range(0, len(bboxIdx)):
                    bbox = bboxs [np. squeeze (bboxIdx[i])]
                    classConfidence = confidences [np. squeeze (bboxIdx[i])]
                    classLabelID = np. squeeze (classLabelIDs [np.squeeze (bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]
                    displayText = "{}:{:.4f}".format(classLabel, classConfidence)
                    x,y,w,h = bbox
                    cv2.rectangle(image, (x,y), (x+w, y+h), color=(255,255,255), thickness=1)
                    cv2.putText(image, displayText, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
                    object_id = f"{classLabel}_{i}"
                    if object_id not in self.object_timestamps:
                        self.object_timestamps[object_id] = time.time()
                        self.object_duration[object_id] = 0
                    else:
                        self.object_duration[object_id] = time.time() - self.object_timestamps[object_id]  
                        if (int(self.object_duration[object_id]*10) % 100 == 0) and int(self.object_duration[object_id]) > 0:
                            print(f"{object_id} has been in frame for {self.object_duration[object_id]} seconds") 
                            shortened_object_id = ''.join(object_id.split('_')[:-1])
                            try:
                                response = self.co.generate( model='command',
                                   prompt=f'You are a strict but sarcastic and passive aggressive mom scolding their child in university that they have left their {shortened_object_id} lying around for {int(self.object_duration[object_id])} seconds already and they should clean it after you took a look into their room.',
                                   max_tokens=38, temperature=0.9, k=0, stop_sequences=[], return_likelihoods='NONE')
                                """
                                url = "https://play.ht/api/v1/convert"

                                payload = {
                                    "content": [response],
                                    "voice": "en-US-JaneNeural",
                                    "title": "yes",
                                    "narrationStyle": "shouting"
                                }
                                headers = {
                                    "accept": "application/json",
                                    "content-type": "application/json",
                                    "AUTHORIZATION": "71dbc92a948c4249aa08daf5a2140965",
                                    "X-USER-ID": "0ONUK3Ak9DaRwdMuFCOj716WHP83"
                                }

                                response = requests.post(url, json=payload, headers=headers)
                                url = f"https://play.ht/api/v1/articleStatus?transcriptionId={response.transcriptionId}"

                                headers = {
                                    "accept": "application/json",
                                    "AUTHORIZATION": "71dbc92a948c4249aa08daf5a2140965",
                                    "X-USER-ID": "0ONUK3Ak9DaRwdMuFCOj716WHP83"
                                }

                                response = requests.get(url, headers=headers)

                                Audio(url=response.audioUrl)
                                print(response.audioUrl)
                                """
                            except:
                                response = f"It's been {int(self.object_duration[object_id])} seconds already. And your {shortened_object_id} is still lying around. Clean it up now!"

                                
                            SpeechThread(self.engine, response.generations[0].text).start()
            #"""
            #if time.time() - last_time >= 10:
            #    print(self.object_duration)
            #    print(sorted([(k,v) for k,v in self.object_duration.items()],key = lambda x: x[1]))
            #    last_time = time.time()
            #"""
            cv2.imshow("Result", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            (success, image) = cap.read()
        cv2.destroyAllWindows()
        print()
        print(self.object_timestamps)
