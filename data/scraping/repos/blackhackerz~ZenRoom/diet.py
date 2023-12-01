# Import Pyrebase
import pyrebase
from flask import Flask, request,jsonify
from flask_cors import CORS
import openai
from helper.openai_api import text_complition
app = Flask(__name__)
openai.api_key = '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'

CORS(app)
# Define your Firebase configuration data
config = {
  "apiKey": "AIzaSyAvYvSqBoQzCUDK2oloq79JhPJGTw1DIUk",
  "authDomain": "dashboard-50078.firebaseapp.com",
  "databaseURL": "https://dashboard-50078-default-rtdb.firebaseio.com",
  "storageBucket": "dashboard-50078.appspot.com"
}

# Initialize Firebase
firebase = pyrebase.initialize_app(config)

# Get a reference to the database service
db = firebase.database()
user='shah'
# Get the list of values under physical act data/kaushal
physical = db.child("physActData").child(user).get().val()
sleep = db.child("sleepData").child(user).get().val()
mood = db.child("moodData").child(user).get().val()
social = db.child("socIntData").child(user).get().val()
stress = db.child("stressData").child(user).get().val()
heart = db.child("heartRateData").child(user).get().val()

# Get the value of the last element of the list
physical = str(physical[-1])
sleep = str(sleep[-1])
mood = str(mood[-1])
social = str(social[-1])
stress = str(stress[-1])
heart = str(heart[-1])


# Print the value
print(physical) # Prints 3
print(sleep) # Prints 3
print(mood) # Prints 3
print(social) # Prints 3
print(stress) # Prints 3
print(heart) # Prints 3

disease='diabetes'
age='35'
gender='male'
location='andheri east, marol,mumbai,india'    
message="sleep : " + sleep + ", heart rate : " + heart + ", mood : "+ mood+ ", physical activity level : " + physical+', social: '+social +  ", stress :" +stress+ ", age: "+ age+ ", gender: "+gender+ ', disease: '+disease+', location : '+location+", prepare a diet plan for me based on these mentioned details , give answer as if you are a doctor and a dietician , prepare diet plan along with calories and all like calories intake ,energy intake ,vitamin intake for breakfast,lunch dinner and snacks etc, also mention the total calorie,vitamins,energy,protien intake at braekfast,lunch and dinner, intotal ,at the end tell a quote regarding mental fitness ,  and a joke to relieve my stress and then tell the user to shift to a vegetarian diet with reasons. NOTE : ANSWER LIKE A DOCTOR AND A DIETICIAN "
print(message)
if message:
    try:
        if len(message.split()) > 300:
            raise ValueError("Input contains more than 300 words. Please try again.")
        chat = openai.Completion.create(engine="text-davinci-003",prompt=message,max_tokens=3896,temperature=0.2)

    except ValueError as e:
        print(f"Error: {e}")
reply = chat.choices[0].text
response_message=f"{reply}"
print(response_message) 