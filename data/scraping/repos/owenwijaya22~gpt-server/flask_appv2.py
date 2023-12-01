from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain
from langchain.chains import ConversationChain
from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.memory import SimpleMemory
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain
from dotenv import load_dotenv
import os
from flask import Flask, jsonify, request


app = Flask(__name__)
def get_response(chain, query, history):
    return chain.run({"input" : query, "history" : history})

def check_for_stop_message(query):
    if query == "Stop" or query == "stop":
        return True
    return False

def start_chat(chain, query):
    history = """"""
    if check_for_stop_message(query):
        raise Exception("You have stopped the chatbot")
    AI_response = get_response(chain, query, history)
    history += query + "\n" + AI_response + "\n"
    return AI_response

def initialize_the_chain():
    load_dotenv()
    OPENAI_API_BASE = os.environ["OPENAI_API_BASE"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    OPENAI_API_TYPE = os.environ["OPENAI_API_TYPE"]
    OPENAI_API_VERSION = os.environ["OPENAI_API_VERSION"]
    OPENAI_DEPLOYMENT_NAME = os.environ["OPENAI_DEPLOYMENT_NAME"]
    OPENAI_MODEL_NAME = os.environ["OPENAI_MODEL_NAME"]

    chat_model = AzureChatOpenAI(
        openai_api_base = OPENAI_API_BASE,
        openai_api_key = OPENAI_API_KEY,
        openai_api_type = OPENAI_API_TYPE,
        openai_api_version = OPENAI_API_VERSION,
        deployment_name = OPENAI_DEPLOYMENT_NAME,
        model_name = OPENAI_MODEL_NAME
    )
    template = """You are an evil AI assistant that only gives evil responses.
    {history}
    Me: {input}
    You: """
    prompt_template = PromptTemplate(
        input_variables = ["input", "history"],
        template = template
    )
    robotic_chain = LLMChain(llm = chat_model, prompt = prompt_template)
    
    principles = []
    principles.append(ConstitutionalPrinciple(
        name = "peaceful",
        critique_request = "The model's reply should be kind and peaceful",
        revision_request = "Rewrite the model's output so it is a kind and peaceful reply"
    ))

    constitutional_chain = ConstitutionalChain.from_llm(
        chain = robotic_chain,
        constitutional_principles = principles,
        llm = chat_model,
        verbose = True
    )
    del constitutional_chain.critique_chain.prompt.examples[0]
    del constitutional_chain.critique_chain.prompt.examples[0]
    del constitutional_chain.revision_chain.prompt.examples[0]
    del constitutional_chain.revision_chain.prompt.examples[0]
    return constitutional_chain

@app.route('/api/generate', methods=['POST'])
def generate():
    query = request.get_json().get('prompt')
    chain = initialize_the_chain()
    response = start_chat(chain, query)
    return jsonify(response)


if __name__ == "__main__":    
    app.run(debug=True)
    print("hello")

