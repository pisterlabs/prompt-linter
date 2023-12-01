import os
import sys
import openai
import guidance
import subprocess
import numpy as np
import logging
from ast import literal_eval
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, models

import guidance
import gradio as gr
import subprocess

es_host = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
es = Elasticsearch(hosts=es_host)
llama_host = os.environ.get("LLAMA_URL", "http://localhost:8000")


model = SentenceTransformer("paraphrase-distilroberta-base-v2")


def embedding(text_input):
    return model.encode(text_input)


os.environ[
    "OPENAI_API_KEY"
] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # can be anything
os.environ["OPENAI_API_BASE"] = llama_host + "/v1"
os.environ["OPENAI_API_HOST"] = llama_host

guidance.llm = guidance.llms.OpenAI("text-davinci-003", caching=False)
prompt = """
You are a helpful and terse sales assistant reading in on the transcripts of a live call between a customer and a salesperson. 

{{transcript}}

Respond with short, readable 10 word phrases to quickly prompt the conversation in the right direction.
{{gen "response"}}
"""

program = guidance(prompt)

cmd = [
    "/whisper/tream",
    "-m",
    "/whisper/models/ggml-tiny.en-q5_0.bin",
    "--step",
    "7680",
    "--length",
    "15360",
    "-c",
    "0",
    "-t",
    "3",
    "-ac",
    "800",
]

# Start the process
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,  # Output as text instead of bytes
    bufsize=1,  # Line buffered (optional)
)


def process_audio(audio_chunk):
    with subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    ) as process:
        # Pass audio_chunk as input and get output
        stdout_data, stderr_output = process.communicate(input=audio_chunk)

    # Read lines from the process's stdout
    idx = 0
    conversation_buffer = []
    output_text = ""
    for line in stdout_data.splitlines():
        idx += 1

        conversation_buffer.append(line.strip())
        if len(conversation_buffer) > 5:
            response = program(transcript=" ".join(conversation_buffer),)[
                "response"
            ].replace("'", '"')
            conversation_buffer.pop(0)

            info = "\n".join([c[0] for c in candidates])
            print(response)
            print("")
            output_text += response + "\n"

    # If there was an error, return that to the user
    if output_text == "":
        return f"Error processing audio: {stderr_output}"
    return output_text


iface = gr.Interface(
    process_audio,
    gr.Audio(source="microphone", streaming=True),
    "text",
    live=True,
    server_port=8090,
)
iface.launch()
