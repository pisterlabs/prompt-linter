import os
from dotenv import load_dotenv
#load_dotenv()
from LangChainModel import Model
from AlpacaModel import AlpacaModel
#from LlamaIndex import IndexModel
from flask import Flask, render_template, request
# Based off of: https://medium.com/@kumaramanjha2901/building-a-chatbot-in-python-using-chatterbot-and-deploying-it-on-web-7a66871e1d9b

# Set the maximum number of tokens to generate in the response
max_tokens = 1024


app = Flask(__name__)
app.static_folder = 'static'

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return runModel(userText)

def runModel(query,gen_ndx=False):
    #if gen_ndx:
    #    AlpacaModel().generateIndex("indexACR")

    AlpacaModel().run(query,"indexACR")
def testModel():
    m = AlpacaModel()
    v_store = m.moreIntelligentIndex("indexACR")
    print(m.self_query(v_store,"What are the imaging recommendations for Chronic Venous Disease"))

if __name__ == "__main__":
    testModel()
    #runModel("A 8 year old male presents with back pain. Which type of imaging is indicated?",False)
