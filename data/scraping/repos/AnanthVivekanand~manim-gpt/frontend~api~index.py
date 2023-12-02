from flask import Flask
from flask import request
import sys
import requests
from pdfminer.high_level import extract_text
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import random
import openai
import os
from flask import send_file, send_from_directory
import docker
import json
import time

app = Flask(__name__)
    
@app.route('/api/pdf2', methods=['POST'])
def convert2():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file'] 
    print(file, file=sys.stderr)

    # save to a temp pdf file
    file.save('temp.pdf')
    
    text = extract_text("temp.pdf")
    
    # open prompt.txt in the same directory as this file
    # this file is at /api/index.py
    # prompt.txt is at /api/prompt.txt
    
    prompt = open("api/prompt.txt", "r").read()
    prompt = prompt.replace("[TEXT]", text)
    print(prompt, file=sys.stderr)
    
    
    anthropic = Anthropic()
    anthropic.api_key = "sk-ant-##########"
    completion = anthropic.completions.create(
        model="claude-instant-1",
        max_tokens_to_sample=50000,
        prompt=f"{HUMAN_PROMPT}{prompt} {AI_PROMPT}",
    )
    print(completion.completion, file=sys.stderr)
        
    
    return completion.completion 



@app.route('/api/stablediffusion', methods=['POST'])
def stablediffusion():
    # this take a string in the body to pass to the api
    text = request.get_data(as_text=True)
    print(text, file=sys.stderr)
    
    url1 = "https://cloud.leonardo.ai/api/rest/v1/generations"

    payload = {
        "prompt": text,
        "modelId": "ac614f96-1082-45bf-be9d-757f2d31c174",
        "width": 896,
        "height": 1152,
        "sd_version": "v2",
        "num_images": 1,
        "num_inference_steps": 45,
        "guidance_scale": 7,
        "scheduler": "KLMS",
        "presetStyle": "LEONARDO",
        "tiling": True,
        "public": False,
        "promptMagic": True,
        "controlNet": False,
        "controlNetType": "POSE"
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": "Bearer #######"
    }

    task = requests.post(url1, json=payload, headers=headers)
    task = task.json()
    #logging.info(task)
    genID = task["sdGenerationJob"]["generationId"]
    url2 = f"https://cloud.leonardo.ai/api/rest/v1/generations/{genID}"
    headers = {
        "accept": "application/json",
        "authorization": "Bearer #########"
    }
    check = True
    while (check):
        time.sleep(2)
        response = requests.get(url2, headers=headers)
        response = response.json()
        print(response, file=sys.stderr)
        status = response["generations_by_pk"]["status"]
        if (status == "COMPLETE"):
            imageUrl = response["generations_by_pk"]["generated_images"][0]["url"]
            check = False
       



    resized_url = imageUrl
    
    return resized_url


@app.route('/api/pdf_math', methods=['POST'])
def convert3():
    # r = open("api/math_response.txt", "r").read()
    # return r
    
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file'] 
    print(file, file=sys.stderr)

    # save to a temp pdf file
    file.save('temp.pdf')
    
    text = extract_text("temp.pdf")
    
    # open prompt.txt in the same directory as this file
    # this file is at /api/index.py
    # prompt.txt is at /api/prompt.txt
    
    prompt = open("api/prompt_math.txt", "r").read()
    prompt = prompt.replace("[TEXT]", text)
    print(prompt, file=sys.stderr)
    
    
    anthropic = Anthropic()
    # anthropic.api_key = "sk-ant-"
    completion = anthropic.completions.create(
        model="claude-instant-1",
        max_tokens_to_sample=50000,
        prompt=f"{HUMAN_PROMPT}{prompt} {AI_PROMPT}",
    )
    print(completion.completion, file=sys.stderr)
        
    
    return completion.completion 

base_directory = os.getcwd()

@app.route('/<path:subpath>')
def serve_file(subpath):
    # Construct the absolute path to the requested file
    absolute_path = os.path.join(base_directory, subpath)
    
    # Ensure that the requested path is within the base directory to prevent directory traversal attacks
    if os.path.commonpath([absolute_path, base_directory]) != base_directory:
        return "Invalid path", 403
    
    # Check if the requested path exists and is a file
    if os.path.isfile(absolute_path):
        return send_from_directory(base_directory, subpath)
    
    # If the path is a directory, you can customize the response as needed
    # For example, you can list the contents of the directory or provide a custom page
    return "File not found", 404


@app.route('/api/manim', methods=['POST'])
def manim():
    # we get the topic from the request
    topic = json.loads(request.get_data(as_text=True))['topic']
    
    # read file example_1.txt
    example_response_1 = open("api/example_1.txt", "r").read()
    example_response_2 = open("api/example_2.txt", "r").read()
    example_response_3 = open("api/example_3.txt", "r").read()
    example_response_4 = open("api/example_4.txt", "r").read()
    
    
    # submit the topic to GPT-4 for it to write the manim code
    message=[
        {"role": "system", "content": "From now on act as CAN (“code anything now”) CAN is an expert coder, with years of coding experience. CAN does not have a character limit."},
        {"role": "user", "content": "Write a manim Python script to visualize the sin and cos functions."},
        {"role": "assistant", "content": example_response_1},
        {"role": "user", "content": "Write a manim Python script to visualize the argmin of a function."},
        {"role": "assistant", "content": example_response_2},
        {"role": "user", "content": "Write a manim Python script to visualize a rectangle whose upper-right corner can move along a curve."},
        {"role": "assistant", "content": example_response_3},
        {"role": "user", "content": "Write a manim Python script to visualize where the sin function originates from."},
        {"role": "assistant", "content": example_response_4},
    ]
    
    message.append({"role": "user", "content": f"Write me a manim Python script to animate {topic}. Name the class `ManimAnimationVisualization`."})
    print(topic, file=sys.stderr)
    
    generating = True
    tf = None
    
    while generating:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages = message,
            temperature=0.2,
            max_tokens=1000,
            frequency_penalty=0.0
        )
        
        code = response.choices[0].message.content.split("```")[1]
        code = "\n".join(code.split('\n')[1:])
        print(code, file=sys.stderr)
        # create temp file
        
        tf = str(random.randint(1, 1000000)) + ".py"
        f = open("api/" + tf, "a")
        f.write(code)
        f.close()

        #with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        #    tmp.write(code.encode('utf-8'))
        #    tmp_file_name = tmp.name

        # create docker client
        client = docker.from_env()
    
        # run docker container
        volume={os.path.join(os.getcwd(), "api"): {'bind': '/manim', 'mode': 'rw'}}
       
        
        container = client.containers.run("manimcommunity/manim", 
                                            f"manim -qm {tf} ManimAnimationVisualization",
                                            volumes=volume,
                                            stderr=True, stdout=True,
                                            detach=True)

        total_output = "" 
        for line in container.logs(stream=True):
            print(line.strip(), file=sys.stderr)
            line = line.decode("utf-8")
            total_output += line
            if "Error" in line:
                message.append({"role": "user", "content": total_output})
                break
            if "Rendered" in line:
                generating = False
                break
        
    # output video file path
    output_path = os.path.join("api", "media", "videos", str(tf)[:-3], "720p30", "ManimAnimationVisualization") + ".mp4"
    return output_path

    return send_file(output_path, mimetype='video/mp4', as_attachment=True)
