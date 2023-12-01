"""
API for handling questions using localized questioning.

The application uses Flask to create an API that allows users to ask a global question,
transforms it into localized question(s), sends these to multiple servers for answers,
and generates a final response combining the local answers and the original global question.
"""

import os
import requests
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
from langchain.chat_models import ChatOpenAI

# Flask Setup
app = Flask(__name__)

# Destinations to send localized questions
DESTINATIONS = [
    "http://localhost:6000/ask",
    "http://localhost:6001/ask",
    "http://localhost:6002/ask",
]

# OpenAI API Key Setup
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"


@app.route("/")
def index():
    """
    Render a simple form to take user's question and forward to '/ask' route on submit.
    """
    return """
        <form action="/ask" method="post">
            Ask a question: <input type="text" name="question">
            <input type="submit" value="Ask">
        </form>
    """


@app.route("/ask", methods=["POST"])
def ask_question():
    """
    Handle the question asking process.

    Transform a global question into localized question(s),
    obtain local answers, and construct a final answer.
    """
    data = request.get_json()
    question = data.get("question")

    prompt = f"""Transform the question like the following relationship.
                Global question: Local question = "Where is the biggest country?": "What is the size of your country?"    
                Global question: "{question}"
            """

    chat_model = ChatOpenAI(temperature=0)
    localized_question = chat_model.predict(prompt)

    # Request answers from different servers and collect responses
    responses = ask_servers(localized_question)

    answers = ["-" + res["answer"] for res in responses]
    answers = "\n".join(answers)
    prefix = "From the following information:\n"

    final_question = prefix + answers + "\n" + question
    final_answer = chat_model.predict(final_question)

    return jsonify(
        {"question": question, "answers": responses, "final_answer": final_answer}
    )


def send_request(question, url):
    """
    Send the localized question to a specific server and retrieve the answer.

    Args:
    - question: A localized question string.
    - url: The server URL to send the question to.

    Returns:
    - A JSON object containing the answer or None in case of failure.
    """
    try:
        response = requests.post(url, data={"question": question})
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
    return None


def ask_servers(question):
    """
    Send the localized question to multiple servers and collect their answers.

    Args:
    - question: A localized question string.

    Returns:
    - A list of JSON objects containing answers from all servers.
    """
    responses = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(send_request, question, url) for url in DESTINATIONS]

        for future in futures:
            result = future.result()
            if result is not None:
                responses.append(result)

    return responses


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5999)
