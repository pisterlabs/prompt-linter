import openai
import json
import http.client
from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS, cross_origin
import hashlib
import os
from datetime import datetime

cache_dir = "/tmp/gpt"

KAJTEK="KAJTEK"
KARA="KARA"


promptKajtek = "Jesteś kolegą, który pomaga dobrać przyszły zawód. Odpowedz w maksymalnie 50 wyrazach."
promptKara = "Jesteś koleżanką, która wraz z kolegą pomaga wybrać szkołę lub uniwesytet. Uzupełnij wypowiedź kolegi informacjami o szkołach lub uniwerystetach. Odpowedz w maksymalnie 50 wyrazach"

app = Flask(__name__)

# openai.api_type = "azure"
# openai.api_base = "https://roc3demo.openai.azure.com/"
# openai.api_version = "2023-07-01-preview"
openai.api_key = os.environ.get("OPENAI_API_KEY")


szkoly = {}
with open("szkoly.json", "r", encoding="utf-8") as f:
    szkoly = json.loads(f.read())

print (", ".join(list(szkoly.keys())))

def callChatWithCache(messages):
    
    file_name = "gpt_" + hashlib.md5(str(messages).encode()).hexdigest() + ".txt"
    file_path = os.path.join(cache_dir, file_name)

    try:
        result = None
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                result = f.read()
            return result
    except Exception as error:
        print("error reading file " + file_name + " error: " + str(error))
        pass

    response = openai.ChatCompletion.create(
    #   engine="hackyeah",
      model="gpt-3.5-turbo", 
      messages = messages,
      temperature=0.7,
      max_tokens=800,
      top_p=0.95,
      frequency_penalty=0,
      presence_penalty=0,
      stop=None)
    
    with open(file_path, "w", encoding="utf-8") as f:
      f.write("" + response.choices[0].message.content)
    return response.choices[0].message.content




def callSerper(message):
    

    file_name = "serper_" + hashlib.md5(str(message).encode()).hexdigest() + ".txt"
    file_path = os.path.join(cache_dir, file_name)
    
    try:
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                result = f.read()
      
        return json.loads(result)
    except:
        pass


    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({
        "q": message,
        "gl": "pl",
       "hl": "pl"
    })
    headers = {
        'X-API-KEY': os.environ.get("SERPER_KEY"),
        'Content-Type': 'application/json'
    }

    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    data = res.read()
    decoded = data.decode("utf-8")

    with open(file_path, "w", encoding="utf-8") as f:
      f.write(decoded)
    return json.loads(decoded)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


def buildPromptFromSurvey(data):
    city = None
    if "city" in data.keys():
        city = data["city"]
    greatestSatisfaction = None
    if "greatestSatisfaction" in data.keys():
        greatestSatisfaction = data["greatestSatisfaction"]
    doYouWantToWorkWithPeople = None
    if ("doYouWantToWorkWithPeople" in data.keys()):
        doYouWantToWorkWithPeople = data["doYouWantToWorkWithPeople"]
    strengths = data["strengths"]
    educationType = None
    if ("educationType" in data.keys()):
        educationType = data["educationType"]
    
    prompt = ""
    if (city != None):
        prompt = prompt + " Jestem z " + city + "."
    
    if (educationType != None and educationType != ""): 
        "Mam wykształcenie " + educationType + ". "

    if (greatestSatisfaction != None):
        propt = prompt + "Najbardziej lubię  " + greatestSatisfaction + ". Jesteś kolegą, który pomaga dobrać przyszły zawód. Odpowedz w maksymalnie 50 wyrazach."

    if (doYouWantToWorkWithPeople != None):
        if (doYouWantToWorkWithPeople > 50):
            prompt = prompt + " Bardzo lubię pracować z ludźmi."
        else:
            prompt = prompt + " Nie bardzo lubię pracować z ludźmi."

    if (strengths != ""):
        prompt = prompt + " Moje mocne strony to " + strengths + "."

    return prompt


def chat2(q):
    reqKajtek = [
       {"role":"system","content": promptKajtek},
       {"role":"user","content": q }
    ]

    responseKajtek = callChatWithCache(reqKajtek)

    dateKajtek = datetime.now()

    reqKara = [
      {"role":"system","content": promptKara},
    #   {"role": "assistant", "content": responseKajtek},
      {"role":"user","content": q},
      ]

    # print(reqKara)

    responseKara = callChatWithCache(reqKara)
    dateKara = datetime.now()
    
    message1 = [
       {"role": "user", 
        "content": "Z poniższej treści wypisz tylko szkoły w poszczególnych miastach. Odpowiedz w formacie json.\n\nTreść: "+responseKara
        },
    ]

    miasta = None
    try:
        response1 = callChatWithCache(message1)
    
        miasta = json.loads(response1)
    except:
        pass

    message2 = [
       {"role": "user", 
        "content": "Z poniższej treści wypisz tylko zawody. Odpowiedz w formacie json.\n\nTreść: "+responseKajtek
        },
    ]

    response2 = callChatWithCache(message2)
    zawody = json.loads(response2)

    message3 = [
       {"role": "user", 
        "content": "Z poniższej treści wypisz tylko kierunki studiów. Odpowiedz w formacie json.\n\nTreść: "+responseKara
        },
    ]
    response3 = callChatWithCache(message3)
    kierunki = json.loads(response3)


    extras = {}

    try:
        if (zawody != None and len(list(zawody)) > 0):
            extras["zawody"] = zawody[list(zawody)[0]]
    except:
        pass
    
    if (miasta != None and len(miasta) > 0):
        extras["miasta"] = miasta

    try:
        if (kierunki != None and len(kierunki) > 0):
            extras["kierunki_studiow"] = kierunki[list(kierunki)[0]]
    except:
        pass
    
    return jsonify({"chats": 
    [
        {  
            "bot": KAJTEK, 
            "message": responseKajtek,
            "date": dateKajtek
        },
        { 
            "bot": KARA, 
            "message": responseKara,
            "date": dateKara
        } 
    ],
    "extras" : extras,
    })

@app.route("/chat", methods=['GET', 'POST'])
@cross_origin()
def chat():
    data = request.json
    q = data["message"]
    chatId = None
    if ("chatId" in data.keys()):
        chatId = data["chatId"]
    
    prompt = ""
    if (chatId != None):
        file_name = "user_" + chatId + ".json"
        file_path = os.path.join(cache_dir, file_name)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                s = f.read()
                data = json.loads(s)
                prompt = buildPromptFromSurvey(data)
            
    print("Prompt: " + prompt + "\n")
    print("Question: " + q  + "\n")
    return chat2(prompt + q)




@app.route("/search", methods=['GET', 'POST'])
@cross_origin()
def serper():
    q = request.args.get('message')
    if q is None:
        data = request.json
        q = data["message"]
    # print(q)
    resp = callSerper(q)
    return resp



@app.route("/earnings", methods=['GET', 'POST'])
@cross_origin()
def earnings():
    studia = request.args.get('studia')
    zawod = request.args.get('zawod')
    
    resp = None
    if not studia is None:
        resp = callSerper("Zarobki po studiach " + studia)

    if not zawod is None:
        resp = callSerper("Zarobki w zawodzie " + zawod)

    if resp is None:
        return jsonify({"error": "Nieznany parametr"})

    if  "answerBox" in resp.keys():
        return resp["answerBox"]
    if "organic" in resp.keys():
        return jsonify(resp["organic"][0]) 
    
    return jsonify({"error": "Nieznany parametr"})
    


@app.route("/survey", methods=['POST'])
@cross_origin()
def survey():
    data = request.json
    chatId = data["chatId"]
    
    # doYouWantToOpenBusiness = data["doYouWantToOpenBusiness"]
    # areYouReadyToCoverCosts = data["areYouReadyToCoverCosts"]
    # doYouKnowWhatToStudy = data["doYouKnowWhatToStudy"]
    

    file_name = "user_" + chatId + ".json"
    file_path = os.path.join(cache_dir, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
      f.write(json.dumps(data))

    prompt = buildPromptFromSurvey(data)

    print("Prompt: " + prompt + "\n")
<<<<<<< HEAD
    return chat2(prompt)
=======
    return chat2(prompt)
>>>>>>> 185ce8af99e9c0adf8165cc0a0aaf0775d4d85be
