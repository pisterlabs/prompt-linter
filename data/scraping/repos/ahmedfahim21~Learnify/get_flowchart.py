from flask import Flask , jsonify, Blueprint, request
from flask_cors import CORS, cross_origin
import openai
from dotenv import load_dotenv
import os
import json


load_dotenv()

os.environ['OPENAI_API'] = os.getenv('OPEN_AI_API')



app = Flask(__name__)
get_flowchart = Blueprint('get_flowchart', __name__)


def talkToGPT(prompt):

    openai.api_key = os.environ['OPENAI_API']

    # Call the OpenAI API to generate a response

    response = openai.ChatCompletion.create(

        model="gpt-3.5-turbo",

        messages=[{

            "role": "system",

            "content": "You are a fun yet knowledgable assistant."

        }, {

            "role": "user",

            "content": prompt

        }],

        temperature=0.6,

        max_tokens=1000)
    
    res = response.choices[0].message.content
    

    # Return the generated modules
    return res


@get_flowchart.route('/get_flowchart',methods=['POST'])
def getchart():
    course = request.get_json()['course_name']
    modules = request.get_json()['modules']
    prompt="Generate a flowchart for the study of "+course+ " with given modules "+",".join(modules)+ " add some other important events if necessary. Make this flowchart in a way that the user gets a basic idea of the course. .Give your response as a JSON in RawNodeDatum interface .Be brief in your instruction and captions, don't add your own comments. Be straight to the point, and never reply things like \"Sure, I can..\" etc."
    return talkToGPT(prompt)