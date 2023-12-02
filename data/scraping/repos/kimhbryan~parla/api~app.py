from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv, find_dotenv
import os
import requests
import cohere
import json
from google.cloud import speech
from google.cloud import translate_v2 as translate
import google.auth
import html


load_dotenv(find_dotenv())
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
cohere = cohere.Client(COHERE_API_KEY)

GOOGLE_CLOUD_API_KEY = os.environ.get("GOOGLE_CLOUD_API_KEY")
GOOGLE_CLOUD_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT_ID")
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS")

credentials, project = google.auth.default()


app = Flask(__name__)
CORS(
    app,
    origins='http://localhost:3000',
    methods=[
        'GET',
        'POST'],
    allow_headers=['Content-Type'])


UPLOAD_FOLDER = "./"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/chat/<topic>", methods=['POST'])
def chat(topic) -> str:
    """Calls cohere's chat feature with custom context

    Parameters
    topic: the topic that the user has selected.

    Returns:
    The generated response from cohere in string format
    """
    if request.method == 'POST':
        message = request.form.get('message')
        skill_context = "The following input was spoken by a person. Rate this person's language skills by responding one of beginner, intermediate, or advanced:"
        skill = generate(skill_context, message)
        message = f"The following input is from a chat about {topic}. Pretend that you are a human agreeing with the user about {topic}." + \
            message + "Respond with an appropriate response based on the chat history and context, as well as the fact that the user has " + skill + " language proficiency." + \
            message + "In short, the out put should be no longer than a 2 sentence long answer."
        chat_history = request.form.get('chat_history', []).split("|")

        chat_history_dicts = []
        for entry in chat_history:
            [role, msg] = entry.split(":")
            chat_history_dicts.append({
                "user_name": role,
                "text": msg
            })

        response = cohere.chat(
            message=message,
            chat_history=chat_history_dicts,
            model="command-light"
        )

        answer = response.text
        return answer


@app.route("/transcribe", methods=['GET', 'POST'])
@cross_origin()
def transcribe_audio():
    if request.method == 'POST':
        f = request.files["blob"]
        l = request.values["lang"]
        client = speech.SpeechClient(
            client_options={
                "api_key": GOOGLE_CLOUD_API_KEY,
                "quota_project_id": GOOGLE_CLOUD_PROJECT_ID}
        )
        content = f.read()
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            sample_rate_hertz=48000,
            audio_channel_count=1,
            language_code=l,
        )
        response = client.recognize(config=config, audio=audio)
        transcript = "".join(
            [result.alternatives[0].transcript for result in response.results])
        transcript = generate(
            "The following sentence may have punctuation and capitalization errors. ONLY correct these errors and nothing else. Do not correct grammar. The sentence is:",
            transcript)
        # transcript = generate(
        #     'This generation tool capitalizes and punctuates samples of text.\n\n',
        #     "nobody likes vegetables I don't like them either\nFixed: Nobody likes vegetables. I don't like them either.\n\nSample: are you sure you are a teacher\nFixed:Are you sure you are a teacher?\n\nSample: my favourite fruits are strawberries pears and watermelons\nFixed: My favourite fruits are strawberries, pears, and watermelons.\n\nSample: " +
        #     transcript +
        #     "\nFixed: ")
        return transcript
    return "Call from GET"


@app.route("/translate", methods=["POST"])
def translate_text():
    if request.method == "POST":
        text = request.values["text"]
        target = request.values["target"]

        client = translate.Client(
            # client_options={"api_key": GOOGLE_CLOUD_API_KEY, "quota_project_id": GOOGLE_CLOUD_PROJECT_ID}
            credentials=credentials
        )
        result = client.translate(text, target_language=target)

        return html.unescape(result["translatedText"])


@app.route("/generate")
def generate(context: str, subject: str) -> str:
    """ Calls cohere's generation feature with a strict context-subject format

    Parameters
    context: The context in which the AI should generate a response
    subject: The actual subject that the AI is considering

    Returns:
    The generated response from cohere in string format
    """
    prompt = context + " " + subject
    response = cohere.generate(
        prompt,
        model="command",
        max_tokens=200
    )

    generated = ""

    for gen in response.generations:
        generated += gen.text

    return generated


@app.route("/analyze")
def analyzetext(text: str, lang: str) -> (bool, str):
    """ Calls languagetoolplus's API to detect grammar errors

    Parameters
    text: The text that should be analyzed
    lang: the language of the text

    Returns:
    A tuple (bool, err) where the first value represents the existence
    of an error, and the second describes it if it exists
    """
    url = "https://api.languagetoolplus.com/v2/check"
    params = {
        "text": text,
        "language": lang,
    }
    res = requests.post(url=url, params=params)

    grammar_err = False
    err_desc = ""
    if len(res.json()["matches"]) > 0:
        grammar_err = True
        for err in res.json()["matches"]:
            err_desc += err["message"]

    return (grammar_err, err_desc)


@app.route("/feedback", methods=["POST"])
def feedback() -> json:
    """Generates feedback for the user based on the transcript of the conversation

    Parameters
    user_transcript: An array consisting of each paragraph the user has spoken
    lang: The language that the user has spoken

    Returns:
    A json dict with recommendations for each erroneous paragraph with the index as the key,
    as well as an overall recommendation corresponding to the key "overall"
    """
    # print(request.form)
    user_transcript = request.form.getlist('user_transcript[]')
    # print(user_transcript)
    lang = request.form.get('lang', "en", str)
    # print(lang)

    grammar_context = "Give a correct version of the following grammatically incorrect paragraph of similar length:"
    confidence_context = "The following input was spoken by a person. Rate this person's speech confidence by responding a number between 0 and 100 and nothing else:"
    recommendations = dict()
    comments = ""

    # add recommendations for specific sections
    for id, seq in enumerate(user_transcript):
        # print((id, seq))
        err = analyzetext(seq, lang)
        # print(err)
        if err[0]:
            grammar_rec = generate(grammar_context, seq)
            # print(grammar_rec)
            confidence_rating = generate(confidence_context, seq)
            recommendations[id] = (grammar_rec, confidence_rating)
            comments += err[1]

    # add overall recommendation
    context = "Give an short, maximum 3 sentence summarized overall recommendation only for a person in order to improve their speaking based on the following obervations about a person's speaking:"

    if len(recommendations) >= 1:
        overall_rec = generate(context, comments)
        recommendations["overall"] = overall_rec

    return json.dumps(recommendations)


if __name__ == '__main__':
    app.run()
