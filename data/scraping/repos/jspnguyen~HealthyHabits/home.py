"""The state for the home page."""
from datetime import datetime


import reflex as rx
import cv2, base64, os, smtplib, time
from cvzone.FaceDetectionModule import FaceDetector
from hume import HumeStreamClient
from hume.models.config import FaceConfig
from PIL.Image import Image 
import PIL
from .base import State
import openai



class HomeState(State):
    """Base state for the app.

    The base state is used to store general vars used throughout the app.
    
    """
    
    form_data: dict = {}
    dat: str
    top_emotions: list[str]
    sender_email = "healthyhabitsCalHacks@gmail.com" 
    startStop: bool = False
    image: Image = PIL.Image.new("RGB", (400,280))
    sessions: list[dict] = []
    
    emailPass = os.environ.get('EMAIL_PASS')
    HUME_KEY = os.environ.get('HUME_KEY')
    openai.api_key = os.environ.get('OPENAI_KEY')

    @rx.background
    async def updateEmotions(self):
        """Handle the form submit."""
        client = HumeStreamClient(self.HUME_KEY)
        config = FaceConfig()
        
        cap = cv2.VideoCapture(0)
        detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)
        curr_frame_count, cooldown_frame_count = 0, 0 
        anger_sum, anxiety_sum, distress_sum = 0, 0, 0
        pain_sum, sadness_sum, tiredness_sum = 0, 0, 0

        anger_sum2, anxiety_sum2, distress_sum2 = 0, 0, 0
        pain_sum2, sadness_sum2, tiredness_sum2 = 0, 0, 0
        total_frames, api_frames = 0, 0
        start_time, end_time = "", ""

        while True:
            if total_frames == 0:
                current_datetime = datetime.now()
                start_time = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
            total_frames += 1

            _, img = cap.read()
            img, bboxs = detector.findFaces(img, draw=False)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            im_pil = PIL.Image.fromarray(img)

            async with self:
                self.image = im_pil
                if not self.startStop:
                    cap.release()
                    current_datetime = datetime.now()
                    end_time = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
                    temp_dict = {anger_sum2 : "Anger", anxiety_sum2 : "Anxiety", distress_sum2 : "Distress", 
                                    pain_sum2 : "Pain", sadness_sum2 : "Sadness", tiredness_sum2 : "Tiredness"}
                    self.sessions.append({"max_emotion" : temp_dict[max(temp_dict)], "min_emotion" : temp_dict[min(temp_dict)],
                                            "mean_anger" : anger_sum2/total_frames, "mean_anxiety" : anxiety_sum2/total_frames,
                                            "mean_distress" : distress_sum2/total_frames, "mean_pain" : pain_sum2/total_frames,
                                            "mean_sadness" : sadness_sum2/total_frames, "mean_tiredness" : tiredness_sum2/total_frames,
                                            "start_time" : start_time, "end_time" : end_time})
                    return
            yield

            if bboxs:
                if api_frames == 30:
                    api_frames = 0
                    _, buffer = cv2.imencode('.jpg', img)
                    jpg_as_text = base64.b64encode(buffer)

                    try:
                        async with client.connect([config]) as socket:
                            result = await socket.send_bytes(jpg_as_text)
                    except:
                        continue

                    try:
                        scores = result['face']['predictions'][0]['emotions'] 
                    except: 
                        continue
                    emotions = sorted(result['face']['predictions'][0]['emotions'], key=lambda x: x['score'], reverse=True)
                    top5_emotions = emotions[:5]
                    async with self:
                        self.top_emotions = [i['name'] for i in top5_emotions]

                    anger_score = scores[4]['score']
                    anxiety_score = scores[5]['score'] 
                    distress_score = scores[19]['score'] 
                    pain_score = scores[34]['score'] 
                    sadness_score = scores[39]['score'] 
                    tiredness_score = scores[46]['score']

                    temp_data = [anger_score, anxiety_score, distress_score, pain_score, sadness_score, tiredness_score] 

                    anger_sum += temp_data[0]
                    anxiety_sum += temp_data[1]
                    distress_sum += temp_data[2]
                    pain_sum += temp_data[3]
                    sadness_sum += temp_data[4]
                    tiredness_sum += temp_data[5]

                    anger_sum2 += temp_data[0]
                    anxiety_sum2 += temp_data[1]
                    distress_sum2 += temp_data[2]
                    pain_sum2 += temp_data[3]
                    sadness_sum2 += temp_data[4]
                    tiredness_sum2 += temp_data[5]
                    
                    curr_frame_count += 1
                    cooldown_frame_count -= 1

                    if curr_frame_count == 15: 
                        anger_mean, anxiety_mean, distress_mean = anger_sum / curr_frame_count, anxiety_sum / curr_frame_count, distress_sum / curr_frame_count
                        pain_mean, sadness_mean, tiredness_mean = pain_sum / curr_frame_count, sadness_sum / curr_frame_count, tiredness_sum / curr_frame_count

                        if cooldown_frame_count <= 0:
                            if anger_mean >= 0.225:
                                yield rx.call_script("const audio = new Audio('/jingle.mp3'); audio.play();")
                                time.sleep(2)
                                yield rx.window_alert("Take a break!")
                                cooldown_frame_count = 3000
                            elif anxiety_mean >= 0.35:
                                yield rx.call_script("const audio = new Audio('/jingle.mp3'); audio.play();")
                                time.sleep(2)
                                yield rx.window_alert("Take a break!")
                                cooldown_frame_count = 3000
                            elif distress_mean >= 0.5:
                                yield rx.call_script("const audio = new Audio('/jingle.mp3'); audio.play();")
                                time.sleep(2)
                                yield rx.window_alert("Take a break!")
                                cooldown_frame_count = 3000
                            elif pain_mean >= 0.2:
                                yield rx.call_script("const audio = new Audio('/jingle.mp3'); audio.play();")
                                time.sleep(2)
                                yield rx.window_alert("Take a break!")
                                cooldown_frame_count = 3000
                            elif sadness_mean >= 0.2:
                                yield rx.call_script("const audio = new Audio('/jingle.mp3'); audio.play();")
                                time.sleep(2)
                                yield rx.window_alert("Take a break!")
                                cooldown_frame_count = 3000
                            elif tiredness_mean >= 0.3:
                                yield rx.call_script("const audio = new Audio('/jingle.mp3'); audio.play();")
                                time.sleep(2)
                                yield rx.window_alert("Take a break!")
                                cooldown_frame_count = 300

                        anger_sum, anxiety_sum, distress_sum = 0, 0, 0
                        pain_sum, sadness_sum, tiredness_sum = 0, 0, 0
                        curr_frame_count = 0
                api_frames += 1

    def setStopRecord(self):
        self.startStop = not self.startStop
        if self.startStop:
            return HomeState.updateEmotions

    def handle_submit(self, form_data: dict):
        """Handle the form submit."""
        for id, value in form_data.items():
            if value != None:
                self.form_data[id] = value
        return [
            rx.set_value(id, "")
            for id, value in self.form_data.items()
    ]

    def alert(self):
        return rx.window_alert("You have updated your profile info!")
    
    def email(self, Temail: str, bodyText: dict):
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(self.sender_email, self.emailPass)
        subject = "Healthy Habits Report"
        body = str(bodyText)
        message = 'Subject: {}\n\n{}'.format(subject, body)
        server.sendmail(self.sender_email, Temail, message)
        server.quit()
        return rx.window_alert("Email Has Been Sent!")
    
    def emailFull(self, Temail: str, bodyText: dict):
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(self.sender_email, self.emailPass)
        subject = "Healthy Habits Report"
        messages = [{"role": "user", "content": "Please format this like a formal email addressed to" + self.form_data['fName'] + " addressed from Healthy Habits Team. Make me a formal email that is properly spaced that uses this data and based on the highest negative emotions, add linked send research papers in a bulletpoint format, online reources, and websites that would help: Make sure this email doesn't contain filler email and uses all the provided data such as the name of the recipient This email should not be a single paragraph block of text. It must be a properly formatted and spaced email please. You cannot start with Dear [recipient], you must start with the actual name of the recipient" + str(bodyText)}]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )
        response_message = response["choices"][0]["message"]["content"]
        response_message = response_message.replace("\n\n", " ")
        response_message = response_message.replace("\n", "")

        body = response_message
        message = 'Subject: {}\n\n{}'.format(subject, body)
        server.sendmail(self.sender_email, Temail, message)
        server.quit()
        return rx.window_alert("Email Has Been Sent!")


    pass