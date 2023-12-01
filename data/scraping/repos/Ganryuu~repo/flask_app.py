from flask import Flask, render_template, request
import openai
import os

app = Flask(__name__)
openai.api_key = os.getenv("sk-")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/answer", methods=["POST"])
def answer():
    topic = request.form["topic"]
    prompt = request.form["prompt"]
    model = "text-davinci-003"
    completions = openai.Completion.create(engine=model, prompt=prompt + " " + topic, max_tokens=1024, n=1,stop=None,temperature=0.7)
    message = completions.choices[0].text
    return render_template("answer.html", response=message) 

@app.route("/download", methods=["POST"])
def download():
    response = request.form["response"]
    if response:
        with open("output.md", "w") as f:
            f.write(response)
        return "Download complete"
    else:
        return "No response available for download"

if __name__ == "__main__":
    app.run(debug=True)
