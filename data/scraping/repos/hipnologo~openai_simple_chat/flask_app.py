# Importing necessary modules: The code imports Flask, render_template, request from Flask, os, and openai modules.
from flask import Flask, render_template, request
import openai
import os

# Creating an instance of the Flask class: The Flask() constructor is used to create an instance of the Flask class and assign it to the app variable.
# Setting OpenAI API key: The OpenAI API key is set using os.getenv() function that retrieves the value of the environment variable "OPENAI_API_KEY".
app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Defining routes: The app has three routes: '/' (the default route), '/answer', and '/download'.
# Index route: The index route ('/') renders a template called "index.html" using the render_template() function.
@app.route("/")
def index():
    return render_template("index.html")

# Answer route: The '/answer' route is used to generate a response based on user input. 
# It takes input values from the web form that are passed to the function via the HTTP POST method. 
# The function then uses OpenAI's API to generate a response using the GPT-3 model with the specified prompt and topic. 
# Finally, it renders the "answer.html" template with the response as a parameter.
@app.route("/answer", methods=["POST"])
def answer():
    topic = request.form["topic"]
    prompt = request.form["prompt"]
    model = "text-davinci-003"
    completions = openai.Completion.create(engine=model, prompt=prompt + " " + topic, max_tokens=1024, n=1,stop=None,temperature=0.7)
    message = completions.choices[0].text
    return render_template("answer.html", response=message) 

# Download route: The '/download' route is used to allow users to download the generated response. 
# If a response is available, it is converted to a markdown file and downloaded. 
# Otherwise, it returns a message stating that no response is available for download.
@app.route("/download", methods=["POST"])
def download():
    response = request.form["response"]
    if response:
        # convert it to a markdown file and allowing the user to download it
        with open("output.md", "w") as f:
            f.write(response)
        return "Download complete"
    else:
        return "No response available for download"

# Running the app: The if __name__ == "__main__": statement ensures that the app is only run if the script is executed directly, and not if it is imported as a module. 
# The app.run() function starts the Flask development server on the local host and enables debug mode.
if __name__ == "__main__":
    app.run(debug=True)
