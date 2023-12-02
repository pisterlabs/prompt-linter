from flask import Flask, request, jsonify
import guidance
import os
import time
import requests
import json
import sys
import logging

port_number = os.getenv('TG_GUIDANCE_PORT_NUMBER', '9555')
guidance_url = f"http://127.0.0.1:{port_number}"
api_endpoint = f"http://127.0.0.1:{port_number}/api/v1/call"

print("guidance_url: ", guidance_url, file=sys.stdout)

timeout_limit = 60 * 60  # Timeout limit set to 60 minutes
time_start = time.time()

# Headers and data for the post request
headers = {
    "Content-Type": "application/json",
}
data = {
    "prompt": "Hello, world!",
    "stop": ["stop_token"],
    "stop_regex": "stop_regex",
    "temperature": 0.5,
    "n": 1,
    "max_tokens": 200,
    "logprobs": 10,
    "top_p": 1.0,
    "echo": False,
    "logit_bias": {},
    "token_healing": True,
    "pattern": "pattern"
}

while True:
    try:
        print("Attempting to connect to guidance...", file=sys.stdout)
        response = requests.post(api_endpoint, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            break
    except requests.ConnectionError:
        print("Connection refused waiting to retry....", file=sys.stdout)
        time.sleep(10)
        if time.time() - time_start > timeout_limit:
            print("Connection timeout", file=sys.stderr)
            raise Exception("Connection timeout")

guidance.llm = guidance.llms.TGWUI(guidance_url)

app = Flask(__name__)
logging.getLogger('flask').addHandler(logging.StreamHandler())

@app.route('/guidance-api', methods=['POST'])
def do_guidance_call():
    try:
        print("Received request at /guidance-api", file=sys.stdout)  # Log at the start of function
        data = request.get_json()

        args = data.get('args', {})
        res_fields = data.get('res_fields', [])
        payload = data.get('payload', "")
        print("payload: ", payload, file=sys.stdout)
        execute_prompt = guidance(payload)

        res = execute_prompt(**args)

        print("res: ", res["response"], file=sys.stdout)
        res_dict = {field: getattr(res, field, None) for field in res_fields}

        return jsonify({"data": res_dict})

    except Exception as e:
        print("An error occurred:", e, file=sys.stderr)  # Log exceptions
        logging.exception(e)
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8181)



