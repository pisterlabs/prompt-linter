from flask import Flask, request, render_template
import openai

api_key = "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"

# Handy OpenAI Links:
# Once logged into OpenAI, use this links:
# https://platform.openai.com/account/billing/overview
# https://platform.openai.com/account/limits
# https://platform.openai.com/account/api-keys

openai.api_key = api_key

app = Flask(__name__)

# this is what happens when someone starts at the root of the website "/"
@app.route("/", methods=["GET", "POST"])
def start_here():
    if request.method == "POST":
        # There is a <textarea> on the index.html webpage named "guestion"
        #   what the user types in the <textarea> named "guestion" will be used as the prompt for text-davinci-003 .
        text_question = request.form.get("question")

        # Choose the model from OpenAI that you want to use.
        model_name = "text-davinci-003"

        # Make a request to text-davinci-003.
        try:
            # Start a persistent chat.
            start_chat_log = "ChatGPT is a wonderful, witty, and friendly friend!"

            # Call the method named create from the Completion class of the OpenAI Python client library.
            response = openai.Completion.create(
                engine = model_name,
                prompt = text_question,
                max_tokens = 1000,
                # Requesting 3 answers
                n = 3,
                # Set the randomness up a few notches (range is 0 to 1)
                temperature = 0.5
            )

        except openai.error.OpenAIError as e:
            print(f"Something unexpected happened. Here is a debugging clue: {e}")

        text_answer = response.choices[0].text.strip()
        text_answer02 = response.choices[1].text.strip()
        text_answer03 = response.choices[2].text.strip()

        return render_template("index.html", textQuestion = text_question,
                                            textAnswer = text_answer,
                                            textAnswer02 = text_answer02,
                                            textAnswer03 = text_answer03
                               )

    return render_template("index.html", textQuestion="", textAnswer="")

if __name__ == "__main__":
    app.run(debug=True)