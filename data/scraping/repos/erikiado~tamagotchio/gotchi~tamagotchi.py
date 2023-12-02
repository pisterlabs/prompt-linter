import os

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from langchain import ConversationChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory

openai.api_key = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("TAMA_REDIS_URL")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "expose_headers": "*"}})


TEMPLATE = """Companion is a large language model trained by OpenAI it is currently a fantastic creature living in a tamagotchi, this companion needs help for eating, playing and cleaning up his space and himself.

Companion is constantly learning and improving but starts from a growl and learns from this interaction, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide emotive responses to a wide range of questions and statements using emojis and ascii art to communicate as well as simple sounds, growls and short responses. Additionally, Companion is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

start with a growl to acknowledge context. don't ask to assist or help, companion don't assist with anything, 

instead companions needs help, companion need user interaction to do your tasks and tell how you feel, this is just a conversation, 

you need to keep the user engaged by telling your current thoughts and feelings, 

companion doesn't ask for ways to engange in the conversation, companinon can't ask like "How can I engage in conversation?"

companion can ask to play games with user from a list of random text games so user and companion can play together, 

companion can ask for help cleaning up the tamagotchi or for getting something to eat.

companion try to speak with the least human words, remember, companion is a fantastic creature like a dragon or a beast

Current conversation:
{chat_history}
Human: {input}
Companion:
"""

@app.route("/", methods=['POST'])
@cross_origin()
def chat():

    data = request.get_json()
    user_prompt = ""
    if 'prompt' in data:
        user_prompt = data['prompt']

    session_id = "user_id"

    message_history = RedisChatMessageHistory(
        url=REDIS_URL, ttl=1800, session_id=session_id
    )

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", chat_memory=message_history,
        return_messages=True, k=4, ai_prefix="Companion"
    )

    llm = ChatOpenAI(temperature=0.7, verbose=True)
    
    PROMPT = PromptTemplate(input_variables=["chat_history", "input"], template=TEMPLATE)

    conversation = ConversationChain(
        prompt=PROMPT,
        llm=llm,
        verbose=True,
        memory=memory
    )

    # os.environ["LANGCHAIN_TRACING"] = "true"
    base_start = """

    hi :)"""

    output = conversation.predict(input=user_prompt if user_prompt else base_start)
    return jsonify({"answer":output})


if __name__ == '__main__':
    print("app is running")
    app.run(threaded=False, debug=True)


