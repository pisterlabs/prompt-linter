from flask import Flask, request, jsonify, render_template
from PIL import Image
from datetime import datetime
import os
import argparse
import generation_functions

from datetime import datetime


import re
import random


import openai

import json

from llama_cpp.llama import Llama, LlamaGrammar

# Put your URI end point:port here for your local inference server (in LM Studio)
openai.api_base = 'http://localhost:1234/v1'
# Put in an empty API Key
openai.api_key = ''

# Adjust the following based on the model type
# Alpaca style prompt format:
prefix = "<|im_start|>"
suffix = "<|im_end|>"

#prefix = "### Instruction:\n"
#suffix = "\n### Response:"

app = Flask(__name__)


class Chatbot:
    def __init__(self, systemPrompt, max_messages=6):
        self.systemPrompt = systemPrompt
        self.messages = [{"role": "system", "content": systemPrompt}]
        self.max_messages = max_messages

        self.defaultOutput = """{
        "summary": "something happens",
        "sceneDescription": "and then something completely different happened",
        "imageDescription": "a picture of something exciting",
        "options": [
            "option 1",
            "option 2",
            "option 3"
        ]       
        }"""


        self.summary = ""

    def chat0(self, text, temperature=0.0):
        message = {"role": "user", "content": f'{prefix}user\n{text}{suffix}\n{prefix}assistant\n'}
        #self.messages.append(message)
        model = "local model"


        print("ABOUT TO DIE",self.messages)

        response = openai.ChatCompletion.create(
            model=model,
            messages=self.messages+[message],
            temperature=temperature,
            #stop=["<|im_end|>"],
            max_tokens=-1
        )

        outputText = response.choices[0].message["content"]

        if not self.validateJson(outputText):

            print("invalid json", outputText)

            # use default output
            outputText = self.defaultOutput

        # add message to self.messages

        messageUser = {"role": "user", "content": f'{prefix}user\n{systemPrompt}{suffix}\n'}

        messageAssistant = {"role": "assistant", "content": f'{prefix}assistant\n{outputText}{suffix}\n'}
        self.messages.append(messageUser)
        self.messages.append(messageAssistant)

        #format messages

        # if messages>max_messages then we need to remove the oldes messages (expect the first which is system prompt)
        if len(self.messages) > self.max_messages:
            new_messages = [self.messages[0]] + \
                self.messages[-self.max_messages+1:]
            self.messages = new_messages

        return outputText
    

    def chat(self, text, temperature=0.0,max_messages=10):
        def format_messages(messages):
            s=""
            for message in messages:
                s+=f'<|im_start|>{message["role"]}\n{message["content"]}<|im_end|>\n'
            s+='<|im_start|>assistant\n'
            return s
        

        #if story summary is empty, use text as story summary
        if self.summary=="":
            self.summary=text
            messageText="""
storySummary: {self.summary}
""".format(self=self)

        else:

            #roll a die to see if we should add a new summary
            dieValue=random.randint(1,20)
            if dieValue==1:
                dieResult="critical fail"
            elif dieValue < args.failure_threshold:
                dieResult="fail"
            elif dieValue < 20:
                dieResult="success"
            else:
                dieResult="critical success"


            messageText="""
storySummary: {self.summary}
Player action: {text}
Action result: {dieResult}
""".format(self=self,text=text,dieResult=dieResult)

        
        message = {"role": "user", "content": messageText}
        self.messages.append(message)

        #trim self.messages if needed
        if len(self.messages) > self.max_messages:
            self.messages = [self.messages[0]]+self.messages[-self.max_messages+1:]

        print("SENDING",format_messages(self.messages))

        f=format_messages(self.messages)

        response = llm(
            f,
            grammar=grammar, max_tokens=350,
            repeat_penalty=1.2,
        )

        print("GOT RESPONSE",response)

        outputText = response['choices'][0]['text']

        if not self.validateJson(outputText):

            print("invalid json", outputText)

            # use default output
            outputText = self.defaultOutput

        # add message to self.messages
        outMessage={"role": "assistant", "content": outputText}

        self.messages.append(outMessage)

        #append summary
        data=json.loads(outputText)
        #if summary doesn't end in ". " then add ".
        if self.summary.endswith(". "):
            pass
        elif self.summary.endswith("."):
            self.summary+=" "
        else:
            self.summary+=". "
        self.summary+=data["summary"]


        return outputText

        
    def validateJson(self, text):
        try:
            json.loads(text)
        except ValueError as e:
            return False

        # assert that json contains image,description, summary and options
        requiredFields = ["imageDescription",
                          "sceneDescription",
                          "summary",
                          "options"]
        for field in requiredFields:
            if field not in text:
                return False

        return True


chatbots = {}


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['user_input']

    # username and story_id
    username = request.json['username']
    story_id = request.json['story_id']

    # check if we have a chatbot for this story_id
    key = f"{username}_{story_id}"
    if key not in chatbots:
        # create new chatbot
        chatbots[key] = Chatbot(systemPrompt)

    chatbot = chatbots[key]

    chat_output = chatbot.chat(user_input)

    # parse json
    chat_output = json.loads(chat_output)

    image = generation_functions.generate_image(
        chat_output["imageDescription"], prompt_suffix=args.prompt_suffix, width=args.image_sizes[0], height=args.image_sizes[1])

    # save image in static/samples
    # replce non alphanumeric in chat_output["image"] with _ and tripm to 100 chars
    image_filename = re.sub(r'\W+', '_', chat_output["imageDescription"])[:100]
    # prepend timestamp
    image_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{image_filename}.png"
    # image path is in /static/samples
    image_path = os.path.join("static", "samples", image_filename)
    image.save(image_path)

    chat_output["image_path"] = image_path
    return jsonify(chat_output)

# render template chat.html


@app.route("/")
def index():
    return render_template("chat.html")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate a story')

    parser.add_argument('--model_id', type=str, default="Lykon/dreamshaper-xl-1-0",
                        help='model id')

    # prompt suffix
    parser.add_argument('--prompt_suffix', type=str, default=", anime, close up headshot, masterpice",
                        help='prompt suffix')

    # image sizes, default = 1024,1024,2048,2048
    parser.add_argument('--image_sizes', type=int, nargs='+', default=[1024, 1024, 2048, 2048],
                        help='image sizes')
    
    #llm model
    parser.add_argument('--llm_model', type=str, default="D:\lmstudio\TheBloke\Mistral-7B-OpenOrca-GGUF\mistral-7b-openorca.Q5_K_M.gguf",
                        help='llm model')
    
    #failure threshold (from 1 to 20)
    parser.add_argument('--failure_threshold', type=int, default=10,
                        help='this is the number that must be rolled on a 20 sided die to succeed')

    args = parser.parse_args()

    systemPrompt = open("story_prompt.txt", "r").read()

    generation_functions.setup(model_id=args.model_id,
                               need_img2img=False,
                               need_ipAdapter=False,
                               need_music=False)
    

    grammar_text = open('grammar.gbnf').read()
    grammar = LlamaGrammar.from_string(grammar_text)
    llm = Llama(args.llm_model,
                n_ctx=4096)

    app.run(debug=True, use_reloader=False)
