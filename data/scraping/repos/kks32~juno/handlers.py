import json
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from jupyter_server.base.handlers import APIHandler
from jupyter_server.utils import url_path_join
import tornado

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(5))
def completion_with_backoff(openai_key, kwargs):
    openai.api_key = openai_key
    return openai.Completion.create(**kwargs)

class RouteHandler(APIHandler):
    @tornado.web.authenticated
    def post(self):
        input_data = self.get_json_body()
        openai_key = input_data.get("openai_key")
        params = input_data.get("params")
        
        if openai_key and params:
            response = completion_with_backoff(openai_key, params)
            self.finish(json.dumps(response))
        else:
            self.set_status(400)
            self.finish({"error": "Missing required fields: openai_key and/or params"})

def setup_handlers(web_app):
    host_pattern = ".*$"
    base_url = web_app.settings["base_url"]
    route_pattern = url_path_join(base_url, "juno", "complete")
    handlers = [(route_pattern, RouteHandler)]
    web_app.add_handlers(host_pattern, handlers)
