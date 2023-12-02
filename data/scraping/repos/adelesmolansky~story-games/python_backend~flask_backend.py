from flask import Flask, Response, request, stream_with_context, json
from glob import glob
import json
import numpy as np
import os
import secrets
import string
import time
import subprocess
import os
import openai
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes
app.config["TEMPLATES_AUTO_RELOAD"] = True

def generate_random_identifier(length=32):
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def generate_response(original_messages):
    print(original_messages)
    assert original_messages[-1]['role'] == 'user'
    user_prompt = original_messages[-1]['content']

    # FILL IN YOUR API_KEY HERE
    openai.api_key = ("")

    response = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[
          # {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt}
      ],
       stream=True
    )

    completion_text = ''
    # iterate through the stream of events
    for event in response:
           print(event)
           if 'role' in event['choices'][0]['delta']:
               # skip
               continue
           if len(event['choices'][0]['delta']) == 0:
               # skip
               continue
           event_text = event['choices'][0]['delta']['content']
           yield event_text
           completion_text += event_text  # append the text

    return completion_text


@app.route('/chat/completions', methods=['POST', 'GET'])
def echo():
    data = request.json
    print(data)
    def event_stream():
        ID = generate_random_identifier()
        return_data = {
          "id": "chatcmpl-{}".format(ID),
          "created": time.time(),
          "object": "chat.completion.chunk",
          "choices": [
              {
                "delta": {
                    "role": "assistant"
                },
                "finish_reason": None,
                "index": 0,
              }
          ],
        }
        yield "data: {}\n\n".format(json.dumps(return_data))

        for total_content in generate_response(data['messages']):
            return_data = {
              "id": "chatcmpl-{}".format(ID),
              "created": time.time(),
              "object": "chat.completion.chunk",
              "choices": [{
                "delta": {
                  "role": "assistant",
                  "content": total_content
                },
                "index": 0,
                "finish_reason": None
              }],
            }
            yield "data: {}\n\n".format(json.dumps(return_data))

        return_data = {
          "id": "chatcmpl-{}".format(ID),
          "created": time.time(),
          "object": "chat.completion.chunk",
          "choices": [
              {
                "delta": {},
                "finish_reason": "stop",
                "index": 0,
              }
          ],
        }
        yield "data: {}\n\n".format(json.dumps(return_data))
        yield "data: [DONE]\n\n"

    return Response(event_stream(), mimetype="text/event-stream")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)