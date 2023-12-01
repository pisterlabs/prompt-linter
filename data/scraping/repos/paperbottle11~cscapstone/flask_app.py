from flask import Flask, send_from_directory, redirect, request, render_template
from pathlib import Path
from base64 import b64decode
import cv2
import numpy as np
import openai
import json
from pydantic import BaseModel
import time

def byte_image_to_numpy(byte_image):
    np_array = np.frombuffer(byte_image, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return img

def show(img, wait=0):
    if type(img) is bytes:
        img = byte_image_to_numpy(img)
    cv2.imshow("img", img)
    cv2.waitKey(wait)

with open("../config.txt", "r") as f:
    api_key = f.read().strip()
    openai.api_key = api_key
    f.close()

def generateImage(prompt, debug=False):
    if debug:
        print("Debug mode is on, skipping image generation")
        return open("test.png", "rb").read()
    
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="256x256",
        response_format="b64_json",
    )

    image_data = b64decode(response["data"][0]["b64_json"])
    return image_data

class WebsiteAIResponse(BaseModel):
    html: str
    image_names: list[str]
    image_prompts: list[str]

def generate(userRequest):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system", "content": "You are a machine that generates a website with HTML."},
            {"role": "user", "content": f"The request is: {userRequest}. Create HTML code with all of the content in English that would be in the request. The output must be valid json text without any unescaped double quotes in the website. Be informative. Use three images. In the backend, make corresponding lists of image file names and of detailed descriptions for each image name. Position these images in the website where they logically make sense. The folder where the images are located is called \"images\". Use bootstrap CSS classes to style the html. Do not link a bootstrap stylesheet."}
        ],
        functions=[
            {
            "name": "create_website",
            "description": "Create a website based on the given request and create image prompts for the images in the website",
            "parameters": WebsiteAIResponse.model_json_schema()
            }
        ],
        function_call={"name": "create_website"}
    )
    with open("json.json", "w") as f:
        f.write(response.choices[0]["message"]["function_call"]["arguments"])
        f.close()
    output = json.loads(response.choices[0]["message"]["function_call"]["arguments"].encode())
    return output["html"], output["image_names"], output["image_prompts"]

app = Flask(__name__, static_folder="static")

@app.route('/', methods=['GET'])
def index():
    return redirect('/home')

global lastQuery
lastQuery = ""

@app.route('/home', methods=['GET'])
def home():
    global lastQuery
    if request.method == 'GET' and "q" in request.args:
        if request.args["q"] == "": return redirect('/home')
        elif request.args["q"] == lastQuery: 
            stylesheet = "journal.css"
            if request.method == 'GET' and "sheet" in request.args: stylesheet=request.args["sheet"]
            return redirect(f"/lastgen?sheet={stylesheet}")
        else: 
            print("Request: " + request.args["q"])
            print("Image Generation: " + "off" if "imagegen" not in request.args else "on")
            startTime = time.time()
            
            print("generating website")
            userRequest = request.args["q"]
            
            startTextTime = time.time()
            
            html, image_names, image_prompts = generate(userRequest)
            
            textTimeElapsed = time.time() - startTextTime
            print("text generation time: " + str(textTimeElapsed))

            insertIdx = html.find("<head>")
            element = "\n<link rel='stylesheet' href='css/{{stylesheet}}'>"
            html = html[:insertIdx+6] + element + html[insertIdx+6:]

            html = html[:insertIdx+6+len(element)] + "\n<link rel='stylesheet' href='css/genstyle.css'>\n<script src='https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js'></script>\n<script src='https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js'></script>\n" + html[insertIdx+6+len(element):]
            
            # Inserts a home button into the generated site
            insertIdx = html.find("<body>")
            html = html[:insertIdx+6] + "\n<a class='button' href='/home'>Home</a>\n<div class='floating-menu'>\n<a class='button' href='/lastgen?sheet=cerulean.css'>Cerulean</a>\n<a class='button' href='/lastgen?sheet=cosmo.css'>Cosmo</a>\n<a class='button' href='/lastgen?sheet=darkly.css'>Darkly</a>\n<a class='button' href='/lastgen?sheet=journal.css'>Journal</a>\n<a class='button' href='/lastgen?sheet=lux.css'>Lux</a>\n<a class='button' href='/lastgen?sheet=quartz.css'>Quartz</a>\n<a class='button' href='/lastgen?sheet=united.css'>United</a>\n</div>\n" + html[insertIdx+6:]
            
            # Writes the generated site to the generated folder
            with open("templates/gen.html", "wb") as f:
                f.write(html.encode())
                f.close()
            
            # Generates images for each image prompt
            print("generating images")
            imageStartTime = time.time()
            
            debug = "imagegen" not in request.args
            for name, prompt in zip(image_names, image_prompts):
                img = generateImage(prompt, debug=debug)
                with open(f"static/images/{name}", "wb") as f:
                    f.write(img)
                    f.close()
            
            imageTimeElapsed = time.time() - imageStartTime
            print("image generation time: " + str(imageTimeElapsed))

            # Save the current query as the last query
            lastQuery = request.args["q"]
            print("serving generated site")
            
            totalTimeElapsed = time.time() - startTime
            print(totalTimeElapsed)
            
            with open("time.txt", "a") as f:
                f.write(f"{totalTimeElapsed},{textTimeElapsed},{imageTimeElapsed},{userRequest},{len(html)},debug:{debug}\n")
                f.close()
            
            stylesheet = "journal.css"
            if request.method == 'GET' and "sheet" in request.args: stylesheet=request.args["sheet"]
            return redirect(f"/lastgen?sheet={stylesheet}")
    return send_from_directory(app.static_folder, path="index.html")

@app.route('/generated/<path:filename>')
def web_gen_assets(filename):
    return send_from_directory("static/generated", filename)

@app.route('/css/<path:filename>')
def css_assets(filename):
    return send_from_directory("static/css", filename)

@app.route('/images/<path:filename>')
def img_gen_assets(filename):
    return send_from_directory("static/images", filename)

@app.route('/lastgen', methods=['GET'])
def lastgen():
    stylesheet = "journal.css"
    if request.method == 'GET' and "sheet" in request.args: stylesheet=request.args["sheet"]
    return render_template("gen.html", stylesheet=stylesheet)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)