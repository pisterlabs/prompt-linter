from flask import Flask, render_template, request
from flask.json import jsonify
import os
from llamaindex.chatbot_agent import create_agent_chain

app = Flask(__name__)
agent = None

# versions of the chatbot (different version for each module)
# ensure that the version name matches the file path of the source documents
versions = ["PC3131", "CS2131"]

@app.route("/chat", methods=['POST'])
def chat():
    global agent
    user_response = request.json

    response = agent(user_response)

    return jsonify(response=response['output'])


@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        version = request.form['version']
        if version not in versions:
            return "nuh-uh", 400
    else:
        # default settings
        version = 'PC3131'

    # do something to load appropriate llamaindex files for the version of the chatbot
    global agent
    agent = create_agent_chain(version)

    return render_template("index.html", versions=versions, selected_version=version)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5601, debug=True)
