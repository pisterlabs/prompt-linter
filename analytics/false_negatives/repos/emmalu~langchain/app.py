from dotenv import load_dotenv
from flask import Flask, request
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
from platform import python_version
from typing import List

print(python_version())
load_dotenv()
OPEN_AI_KEY = os.getenv("OPENAI_API_KEY")
chat = ChatOpenAI(temperature=0.5)

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return """
        <strong>Hello, Hack for good!</strong>
        <br />
        Please use a valid endpoint.
    """


@app.route("/tour", methods=["GET", "POST"])
def make_tour():
    # # Our results data structure.
    # class TourInfo(BaseModel):
    #     location_name: str
    #     category: str
    #     geolocation: str
    #     story_description: str

    # location = "Charleston, SC"
    location = "Philadelphia, PA"

    chat_prompt = f"""
        You are an expert in planning walking tours around {location}.
        """
    system_prompt = SystemMessagePromptTemplate.from_template(chat_prompt)
    """ chat_params = {
        "interests": ["architecture", "food"],
        "budget": 500,
        "duration": 4,
        "distance": 5,
        "start": "morning",
    }
    interests = ", ".join(chat_params["interests"])
    budget = chat_params["budget"]
    duration = chat_params["duration"]
    distance = chat_params["distance"]
    time_of_day = chat_params["start"] """

    interests = request.form["interests"].split(", ")
    if len(interests) > 1:
        interests = f"""{", ".join(interests[:-1])} and {interests[-1]}"""
    budget = request.form["budget"]
    duration = request.form["duration"]
    distance = request.form["distance"]
    time_of_day = request.form["start"]

    # check if any of the parameters are empty
    if not interests or not budget or not duration or not distance or not time_of_day:
        # return error
        return "Please fill out all fields"

    human_template = f"""
        I am interested in {interests}. I have {duration} hours and would like to walk no more than {distance} miles. My budget is {budget} dollars. I want to start in the {time_of_day}. Please give me a list of locations for a {location} walking tour based on the previous parameters. With each location, provide a recommended start time recommended time at the location, a category, a fun, detailed story about the location, and its geolocation. Present each location as an enthusiastic tour guide. Format the results into a json response complete with location name, category, suggested_start_time, suggested_visit_duration, story and geolocation field.
        """
    # print("HUMAN ASK", human_template)
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(
        input_language="English", output_language="English", max_tokens=100
    )

    return response
