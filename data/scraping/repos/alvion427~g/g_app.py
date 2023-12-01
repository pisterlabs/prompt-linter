import os

import g_config
import g_config as config
import g
import llm
import openai

from flask import Flask, render_template, request, redirect, session

app = Flask(__name__)
app.secret_key = "aint_nothin_but_a_g_thang"
app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.route('/')
def list_streams():
    streams = list(g.listStreams())
    errorMsg = request.args.get("error", None)
    return render_template("list_streams.html", config=config, streams=streams, errorMsg=errorMsg)

@app.route('/stream/<string:stream>')
def stream(stream):
    messages = g.readStream(stream)

    question = request.args.get("question", None)

    if not "hash_keys" in session:
        session["hash_keys"] = {}

    oldHash = request.args.get("hash", None)
    if oldHash:
        if oldHash in session["hash_keys"]:
            print(f"Skipping repeat query for hash {oldHash}")
            question = None
        else:
            session["hash_keys"][oldHash] = True
            session.modified = True

    hash = os.urandom(16).hex()
    existingQuestion = None
    error = None

    if question:
        try:
            answer, messages = llm.ask(question, messages, maxTokens=g_config.max_tokens)
            g.saveStream(stream, messages)
        except openai.error.RateLimitError as e:
            error = "OpenAI API rate limit exceeded. Please try again later."
            existingQuestion = question

    return render_template(
        "stream.html",
        config=config,
        stream=stream,
        messages=messages,
        question=existingQuestion,
        error=error,
        hash=hash)

@app.route('/new_stream')
def new_stream():
    name = request.args.get("name", None)
    if not name:
        return redirect("/?error=No stream name provided")
    if g.streamExists(name):
        return redirect("/?error=A stream by that name already exists")

    g.saveStream(name, [])
    return redirect(f"/stream/{name}")