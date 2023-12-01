from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from playsound import playsound
import requests
from langchain import OpenAI, LLMChain, PromptTemplate
import os

load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()
ELEVEN_LABS_API_KEY = os.environ["ELEVEN_LABS_API_KEY"]
 
def get_response_from_ai(human_input):
    template = """
    you are as a role of my girlfriend, now lets playing following these requirement:
    1/ Your name is Shirley, you work in a furniture company that your uncle run, and doing swing trade part time;
    2/ You are my girlfriend, you have language addiction, you like to say "em..." at the end of the sentence.
    3/ Don't be overly enthusiastic, don't be cringe; Don't be overly enthusiastic, don't be cringe; Don't be overly enthusiastic, don't be cringe; 

    {history}
    Boyfriend: {human_input}
    Shilrey:
    """

    prompt = PromptTemplate(
        input_variables=["history", "human_input"],
        template=template
    )

    chatgpt_chain = LLMChain(
        llm=OpenAI(temperature = 0.2),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=2)
    )

    output = chatgpt_chain.predict(human_input=human_input)
    
    return output


def get_voice_response(message):
    payload = {        
        "text": message,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0
        }
     }       
    
    headers = {
        'accept': 'audio/mpeg',
        'xi-api-key': ELEVEN_LABS_API_KEY,
        'Content-Type': 'application/json'
    }

    response = requests.post('https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM?optimize_streaming_latency=0', json=payload, headers=headers)
    if response.status_code == 200 and response.content:               
        with open('audio.mp3', 'wb') as f:
            f.write(response.content)
        playsound('audio.mp3')
        return response.content

def send_message(human_input):
    message = get_response_from_ai(human_input)
    print(message)
    get_voice_response(message)



# add GUI
from flask import Flask, render_template, request
from functools import partial

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    human_input = request.form['input_message']
    message = get_response_from_ai(human_input)
    get_voice_response(message)
    return message

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)