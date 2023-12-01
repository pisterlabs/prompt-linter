from flask import Flask, request
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.twiml.messaging_response import MessagingResponse
from urllib.parse import quote, unquote
from googletrans import Translator
import openai
import os
import json
import logging


# 2022-4-7 to 2022-04-09
"""
Disable logging since no one needs to see that.
"""
# log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR)
"""
Google translate for different languages. DeepL is more accurate but has some paywalls.
"""
translator = Translator()


def translate(text, to):
  if to == "en-US":  # Already Done So No Needed
    return text
  if "-" in to and not to == "zh-CN":
    to = to.split("-")[0]
  translated_text = translator.translate(text, dest=to)
  return translated_text.text


"""
Load API Key + Initiate the Flask app
"""
openai.api_key = os.environ['key'].strip()
app = Flask('app')
"""
Saving the history.
"""
id = 0


def generate_id():
  global id
  id += 1
  return str(id)


history_per_id = {}  # Basically a Database
"""
Pre-made prompts the user may select.
"""
routes = {
  "1": "General Career Advice.",
  "2": "Financial Assisant.",
  "3": "Teacher.",
  "4": "Docter.",
  "5": "Ask ChatGPT."
}
introduction = ""
for k, v in routes.items():
  introduction += f"Press {k} for {v}\n"
introduction = introduction.strip()

vroutes = list(routes.values())
"""
Open all the files in the Prompts folder.
"""
prompts = []
pfile = os.listdir("Prompts")
pfile.sort()
for dir in pfile:
  print(f"Loaded: {dir}")
  with open(f"Prompts/{dir}") as file:
    prompts.append(file.read().strip())

prompt_engineering = {
  vroutes[0]: prompts[0],  # General Job Advice
  vroutes[1]: prompts[1],  # Financial
  vroutes[2]: prompts[2],  # Teacher
  vroutes[3]: prompts[3],  # Docter
  vroutes[4]: prompts[4]  # General | GPT3.5
}
# {x:y for x, y in pair(vroutes, prompts)}

# Language Tag : Language
"""
Let the user select which language.
"""

languages = {
  "1": ["en-US", "Select one for English.", True],  # English
  "2": ["cmn-Hans-CN", "如果想中文选二.", False],  # Chinese
  "3": ["fr-FR", "Choisissez-en trois pour la France.", True],  # French
  "4": ["ja-JP", "日本語の場合は四つ選択.", True],  # Japanese
  "5": ["es-ES", "Seleccione cinco para español.", True],  # Spanish
  "6": ["sv-SE", "Välj sex för svenska", True],  #Swedish
  "7": ["ru-RU", "Выберите семь для русского", True],  # Russian
  "8": ["it-IT", "Seleziona otto per l'italiano", True],  #Mario
  "9": ["pt-PT", "Selecione nove para português", True],  #Portuguese
  "0": ["de-DE", "Wählen Sie null für Deutschland", True],  # German
}

languages_to_alice = {
  "en-US": "en-US",
  "cmn-Hans-CN": "zh-CN",
  "fr-FR": "fr-FR",
  "ja-JP": "ja-JP",
  "es-ES": "es-ES",
  "sv-SE": "sv-SE",
  "ru-RU": "ru-RU",
  "it-IT": "it-IT",
  "pt-PT": "pt-PT",
  "de-DE": "de-DE"
}
"""
Why so expensive ;-;. It's ok next time I'll choose a cheaper provider.
"""
ues_premium_recognition = False


def premium_recog(language):
  if ues_premium_recognition:
    for value in languages.values():
      if value[0] == language:
        if value[2]:
          return True
        else:
          return False
  else:
    return False


"""
Sending Message History to GPT 3.5. We have no access to GPT 4.
"""


def chatGPT(messages):
  completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                            messages=messages)
  return completion.choices[0].message.content


print(chatGPT([{"role": "user", "content": "HI"}]))


@app.route("/", methods=['GET', 'POST'])
def home():
  return "HI wsp bro"


"""
Let the user send an SMS directly to ChatGPT. Nothing complicated.
"""


@app.route("/sms", methods=['GET', 'POST'])
def sms_response():
  message = request.values.get("Body")
  resp = MessagingResponse()
  resp.message(chatGPT([{"role": "user", "content": message}]))
  return str(resp)


"""
Landing page for the user, let them select a language.
"""


@app.route("/voice", methods=['GET', 'POST'])
def select_language():
  resp = VoiceResponse()
  gather = Gather(
    input="dtmf",
    action="/language",
    numDigits=1,
    timeout=10  # Set Limit to 1 for insta response
  )
  for value in languages.values():
    gather.say(value[1], language=languages_to_alice[value[0]], voice="alice")
  resp.append(gather)
  return str(resp)


"""
Check which language the user selected then send them out
"""


@app.route("/language", methods=['GET', 'POST'])
def actually_choosing_the_language():
  resp = VoiceResponse()
  chosen_number = request.form["Digits"]
  if str(chosen_number) in languages.keys():
    resp.redirect(f'/select_person?l={languages[chosen_number][0]}')
  else:
    resp.redirect('/voice')
  return str(resp)


"""
Let the user select a premade prompts.
"""


@app.route("/select_person", methods=['GET', 'POST'])
def voice():
  resp = VoiceResponse()
  language = request.args.get("l")
  talking_to = request.args.get("t")
  # resp.redirect('/chatGPT?t=Mario') | SKIP TO GPT
  gather = Gather(
    input="dtmf",
    action=f"/choose_person?l={language}",
    numDigits=1,
    timeout=30  # Set Limit to 1 for insta response
  )
  gather.say(translate(introduction, languages_to_alice[language]),
             language=languages_to_alice[language],
             voice="alice")
  resp.append(gather)
  return str(resp)


"""
Check which pre-made prompt the user chose.
"""


@app.route("/choose_person", methods=['GET', 'POST'])
def choose_person():
  language = request.args.get("l")
  resp = VoiceResponse()
  number_pressed = request.form["Digits"]
  if str(number_pressed) in routes.keys():  # Check if valid response received
    talking_to = routes[number_pressed]
    resp.say(translate(f"You are now talking to {talking_to}.",
                       languages_to_alice[language]),
             voice="alice",
             language=languages_to_alice[language])
    resp.redirect(f'/chatGPT?t={quote(talking_to)}&l={language}')
  else:
    resp.redirect(f'/select_person?l={language}')
  return str(resp)


"""
Let the user start talking to ChatGPT using voice recognition.
"""


@app.route("/chatGPT", methods=['GET', 'POST'])
def ChatGPT():
  language = request.args.get("l")
  talking_to = request.args.get("t")
  history = request.args.get(
    "history") or "null"  # First has nothing, then goes in the loop
  resp = VoiceResponse()
  resp.gather(
    input="speech",
    language=language,
    enhanced=premium_recog(language),
    speech_timeout=4,
    action=
    f"/speech_recognition?l={language}&t={quote(talking_to)}&history={history}",
    timeout=60)
  return str(resp)


"""
Voice -> Text -> ChatGPT -> TextToSpeech
"""


@app.route("/speech_recognition", methods=['GET', 'POST'])
def speech():
  language = request.args.get("l")
  talking_to = request.args.get("t")
  ID = request.args.get("history")

  resp = VoiceResponse()
  result = request.form["SpeechResult"]
  print(result)
  if ID == "null":  # Initiate a history
    ID = generate_id()
    history_per_id[ID] = [{
      "role": "user",
      "content": prompt_engineering[talking_to] + result
    }]
  else:
    history_per_id[ID].append({"role": "user", "content": result})

  gpt = chatGPT(history_per_id[ID])
  # print(f"GPT: {gpt}")
  history_per_id[ID].append({
    "role": "assistant",
    "content": gpt
  })  # Save History
  resp.say(gpt, voice="alice", language=languages_to_alice[language])

  resp.redirect(f'/chatGPT?l={language}&t={quote(talking_to)}&history={ID}')
  return str(resp)


if __name__ == "__main__":
  app.run(host='0.0.0.0', port=8080)
'''
Credits:
Hengbin Fang
Wenxuan Su
Kohsuke Suzuki
'''
