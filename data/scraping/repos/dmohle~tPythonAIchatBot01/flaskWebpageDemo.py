from flask import Flask, request, render_template
import openai

api_key = "sk-l6bvz... your API key goes here ...CFyFS5t"

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
            # Call the method named create from the Completion class of the OpenAI Python client library.
            response = openai.Completion.create(
                engine = model_name,
                prompt = text_question,
                max_tokens = 1000
            )

        except openai.error.OpenAIError as e:
            print(f"Something unexpected happened. Here is a debugging clue: {e}")

        text_answer = response.choices[0].text.strip()

        return render_template("index.html", textQuestion=text_question, textAnswer=text_answer)
    return render_template("index.html", textQuestion="", textAnswer="")

if __name__ == "__main__":
    app.run(debug=True)
