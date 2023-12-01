from flask import Flask, render_template, request
import pickle
from dotenv import dotenv_values

import openai
openai.api_key = dotenv_values(".env")["OPEN_AI_KEY"]
# audio_file = open("uploadmp3/run.mp3", "rb")
# transcript = openai.Audio.transcribe("whisper-1", audio_file)

# print(transcript)










app = Flask(__name__)
import requests

# url = "https://app.amattendees.com/v2/getdata.php?query=Anna&intent=all"

# payload={}
# headers = {
#   'Cookie': 'PHPSESSID=921020764f422f4cb7a1a69853e08186'
# }

# response = requests.request("GET", url, headers=headers, data=payload)

# pickle.dump(response, open("page1.pickle", "wb"))

# print(response.text)

# response = pickle.load(open("page1.pickle", "rb"))
# print(response.text)
# tiger = response.text


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part in the request', 400
    print("**************************************************************************",request.files['file'])

    file = request.files['file']

    
    file.save('uploadmp3/'+ file.filename)
    audio_file = open('uploadmp3/'+ file.filename, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print(transcript["text"])

    # return render_template("Victory.html", jsonlist = "cool")
    # return transcript["text"], 200
    return transcript["text"]


@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/voicetranscriptions')
def react():
    return render_template("react.html")

@app.route('/seopredictor')
def seopredictor():
    return render_template("seopredictor.html")

@app.route('/leetcode')
def leetcode():
    return render_template("leet-code-index.html")

@app.route('/leetcode/1')
def num_1():
    return render_template("/leetcode/1.html")

@app.route('/test')
def test():
    return render_template("test.html")

@app.route('/shawn' , methods=['POST', "GET"])
def shawn():
        if request.method == 'GET':
        #  return render_template("Victory.html", jsonlist = response.json())
            # return render_template("Victory.html", jsonlist = jsonify(response.json()))
            # return render_template("Victory.html", jsonlist= response.text)
            # return render_template("Victory.html", jsonlist = Markup(tiger))
            # return render_template("Victory.html", jsonlist = response.json())
            return render_template("Victory.html", jsonlist = "cool")
        
        
@app.route('/shawnn' , methods=['POST', "GET"])
def shawnn():
        payload = request.form['payload']


        url = f"https://app.amattendees.com/v2/getdata.php?query={payload}&intent=all"

        payload={}
        headers = {
        'Cookie': 'PHPSESSID=921020764f422f4cb7a1a69853e08186'
        }

        responsee = requests.request("GET", url, headers=headers, data=payload)
        

        # responsee = response


        output = ""
        for x in responsee.json():
             output += "\n\n\n" + f'{x}'  + 'next'
            
        return render_template("Victory.html", jsonlist = responsee.json())


        # return render_template('Victory.html', jsonlist = output)


