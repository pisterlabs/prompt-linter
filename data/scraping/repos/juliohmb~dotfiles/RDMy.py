# openai.api_key = "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"
from flask import Flask, render_template
import openai

app = Flask(__name__)

openai.api_key = "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"

@app.route("/")
def index():
  return render_template("index.html")

@app.route("/response", methods=["POST"])
def response():
  prompt = request.form["prompt"]
  response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt,
    max_tokens=1024,
    temperature=0.5,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  ).get("choices")[0].get("text")
  return response

if __name__ == "__main__":
  app.run()
