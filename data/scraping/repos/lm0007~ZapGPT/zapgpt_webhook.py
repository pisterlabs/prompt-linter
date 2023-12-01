import os
import openai
from heyoo import WhatsApp
from dotenv import load_dotenv
from flask import Flask, request, make_response
from threading import Thread

load_dotenv()

TOKEN = os.getenv('TOKEN')
ID = os.getenv('ID')
OPENAI_TOKEN = os.getenv('OPENAI_TOKEN')

openai.api_key = OPENAI_TOKEN
model_engine = "text-davinci-003"

# Initialize Flask App
app = Flask(__name__)

messenger = WhatsApp(TOKEN, ID)
VERIFY_TOKEN = os.getenv('VERIFY_TOKEN')


def resposta(message, mobile):

    message = f"Human: {message}\nIA: "

    try:

        completion = openai.Completion.create(
            engine=model_engine,
            prompt=message,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.9,
        )

        resposta = completion["choices"][0]["text"]

        messenger.send_message(resposta, mobile)

    except Exception as err:
        messenger.send_message("Ops! Falhei em comunicar com o GPT, tente novamente...", mobile)


@app.route("/", methods=["GET", "POST"])
def hook():
    if request.method == "GET":

        if request.args.get("hub.verify_token") == VERIFY_TOKEN:
            response = make_response(request.args.get("hub.challenge"), 200)
            response.mimetype = "text/plain"
            return response

        return "Invalid verification token"

    data = request.get_json()
    changed_field = messenger.changed_field(data)

    if changed_field == "messages":

        new_message = messenger.get_mobile(data)

        if new_message:

            mobile = messenger.get_mobile(data)
            name = messenger.get_name(data)
            message_type = messenger.get_message_type(data)

            if message_type == "text":

                message = messenger.get_message(data)
                name = messenger.get_name(data)

                Thread(target=resposta, args=(message, mobile)).start()

            else:
                messenger.send_message(f"Foi mal! Trabalhamos apenas com mensagens de texto...", mobile)

        else:

            delivery = messenger.get_delivery(data)

            if delivery:
                print(f"Message : {delivery}")
            else:
                print("No new message")

    return "OK", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.getenv('PORT'))
