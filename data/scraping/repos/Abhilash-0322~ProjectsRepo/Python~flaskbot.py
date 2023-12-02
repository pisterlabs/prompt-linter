from flask import Flask, render_template, request, jsonify
import openai

app = Flask(__name__)

openai.api_key = "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"  # Replace with your OpenAI API key

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form.get("user_input")

    # Use the OpenAI GPT-3.5 model to generate a response
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=f"act as logical scientist\n{user_input}",
        max_tokens=50  # Adjust the max tokens as needed
    )

    bot_response = response.choices[0].text
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)