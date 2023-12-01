import os

from flask import Flask, render_template
import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")


@app.route("/")
@app.route("/<place>")
def hello(place="Berlin"):
    # higher temperature leads to more variation, randomness and creativity
    # temperature between 0.7 and 0.9 is most commonly used if you want to experiment and create many variations quickly
    llm = OpenAI(temperature=0.9)
    # https://python.langchain.com/en/latest/modules/prompts/prompt_templates.html
    prompt = PromptTemplate(
        input_variables=["place"],  # list of variables
        template="What are the 3 best places to eat in {place}?",  # prompt
    )
    question = prompt.format(place=place)
    # split() is used to split the items into a list. The llm response will look like:
    # "\n\n1. <first item>.\n\n2. <second item>..."
    return render_template("hello.html", place=place, answer=llm(question).split("\n\n"))
