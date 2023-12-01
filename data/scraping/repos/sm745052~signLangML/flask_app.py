from flask_socketio import SocketIO, send
from flask import Flask, render_template, Response, request
import cv2
import os
import numpy as np
from coordinates import handDetector
import time
import tensorflow as tf
import numpy as np
import pickle
import keyboard
import statistics
import pyttsx3
import openai
from play import play

#environment variables
os.environ['backspace']="0"
os.environ['enter']="0"






#initialize pyttsx3 engine for sound output
engine = pyttsx3.init() 

#initialize the play class
play = play()





#initialize for sentence and openai
my_sentence = ''
prediction_stack = ['-' for i in range(5)]
blackboard = []
l = 10
openai.api_key = "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"
os.environ['lastword'] = ''



#load the stored labels
with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)




#load the stored model
model = tf.keras.models.load_model("./my_model")




#function for making sentence using openai
def make_sentence():
    global blackboard
    response = openai.Completion.create(
            engine="text-davinci-002",
            prompt="Correct this to standard English: {}".format(' '.join(list(filter(('-').__ne__, blackboard)))+'.'),
            temperature=0,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
    sentence = response["choices"][0]["text"]
    print(sentence)
    return sentence





#function for displaying the sentence
def display():
    global prediction_stack
    global blackboard
    global l
    global my_sentence
    try:
        if(len(prediction_stack) > l):
            word = statistics.mode(prediction_stack[-l:])
            if(len(blackboard) == 0 or word!=blackboard[-1]):
                blackboard.append(word)
                my_sentence = ' '.join(list(filter(('-').__ne__, blackboard)))
                os.environ['lastword'] = word
                socketio.send(my_sentence)
    except:
        pass
    print(blackboard)







def predict_show(img):
    global prediction_stack, l, labels, model, blackboard
    if os.getenv('enter')=="1":
        socketio.send("$"+make_sentence())
        os.environ["enter"]="0"
    if os.getenv('backspace')=="1":
        os.environ['backspace']="0"
        if(len(blackboard)==0):return
        w = blackboard.pop()
        prediction_stack = ['-' for i in range(5)]
        my_sentence = ' '.join(list(filter(('-').__ne__, blackboard)))
        socketio.send(my_sentence)
    pTime = 0
    img = detector.findcoords(img)
    lmlist = detector.findPosition(img)
    if(lmlist[0] != 0):         #deal with only when hands detected          
        y = model.predict(np.array(lmlist).reshape(1, -1))
        result = labels[np.argmax(y.flatten())]
        prediction_stack.append(result)
        display()
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    return img








global capture,rec_frame, switch, detect, rec, out 
switch=0
detect=0

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass
#Load pretrained face detection model    


#instatiate flask app  
app = Flask(__name__, template_folder='./templates')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins='*')


camera = cv2.VideoCapture(0)

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)


def detect_landmarks(frame):
    global detector
    frame = predict_show(frame)
    return frame
 

def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        if success:
            if(detect):                
                frame= detect_landmarks(frame)
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass  
        else:
            pass








@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_1')
def video_feed_1():
    return Response(play.recognize(), mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route("/audio_feed")
# def audio_feed():
#     """Audio streaming route. Put this in the src attribute of an audio tag."""
#     return Response(generateAudio(), mimetype="audio/x-wav")

@socketio.on('message')
def handleMessage(msg):
    print('Message: ' + msg)
    if(msg=="Backspace"): os.environ['backspace']="1"
    if(msg=="Enter"): os.environ['enter']="1"
    

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if  request.form.get('detect') == 'Detect':
            global detect
            detect=not detect
        elif  request.form.get('stop') == 'Stop/Start':
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
            else:
                camera = cv2.VideoCapture(0)
                switch=1
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')










if __name__ == '__main__':
    detector = handDetector()
    app.run()
    
camera.release()
cv2.destroyAllWindows()