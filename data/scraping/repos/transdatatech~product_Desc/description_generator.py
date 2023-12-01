from dotenv import load_dotenv
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import os
import json
import openai
from langchain.llms import OpenAI
from guard_prompt_manual import rail_spec
from jinja2 import Environment, BaseLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import time
import logging
import sys
import warnings
from datetime import datetime
from flask_log_request_id import RequestID, RequestIDLogFilter
warnings.filterwarnings("ignore", category=UserWarning,
                        module="langchain.llms.openai")


class StructuredMessage(object):
    def __init__(self, message, **kwargs):
        self.message = message
        self.kwargs = kwargs

    def __str__(self):
        return '%s >>> %s' % (self.message, json.dumps(self.kwargs))


m = StructuredMessage   # optional, to improve readability

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.6))

app = Flask(__name__)
RequestID(app)
CORS(app)

handler = logging.StreamHandler(sys.stdout)
handler.addFilter(RequestIDLogFilter())
handler.setFormatter(
    logging.Formatter("[%(levelname)s][%(request_id)s] - %(message)s")
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger = logging.getLogger(__name__)


@app.before_request
def start_timer():
    g.start_time = time.time()


@app.after_request
def log_request(response):
    duration = round(time.time() - g.start_time, 2)
    dt = datetime.fromtimestamp(time.time())
    timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')

    ip_address = request.headers.get('X-Forwarded-For', request.remote_addr)
    log_details = {
        "method": request.method,
        "path": request.path,
        "status": response.status_code,
        "duration": duration,
        "time": timestamp,
        "ip": ip_address,
        'request': 'completed'
    }

    logger.info(json.dumps(log_details))

    return response


@app.route('/api/description', methods=['POST'])
def create_description():
    logger.info(m('request.json', request_body=request.json))

    data = request.json
    shop_notes = data['data']
    product = shop_notes['attributes'].get('product_type')
    meta_info = shop_notes['attributes'].get('meta_info')

    rail_spec_prompt = rail_spec
    template = Environment(loader=BaseLoader).from_string(rail_spec_prompt)
    rail_spec_prompt = template.render(
        product=product,
        meta_info=meta_info,
        shop_notes=shop_notes,
    )

    model = OpenAI(
        temperature=TEMPERATURE,
        openai_api_key=openai.api_key,
        model_name='gpt-4',
        streaming=False,
        callbacks=[StreamingStdOutCallbackHandler()]
    )

    logger.info(m('prompt', prompt=rail_spec_prompt))

    response = {
        "data": {
            "type": "description",
            "attributes": {
                "body": model(rail_spec_prompt)
            }
        }
    }
    logger.info(m('response.json', response=response))

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
