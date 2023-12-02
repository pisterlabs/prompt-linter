import configparser
import openai
from flask.views import MethodView
from flask_smorest import Blueprint, abort

blp = Blueprint("Resume", __name__, description="CSS Art Generator")

AUTH_FILE = "/app/.keys"

config = configparser.RawConfigParser()
config.read(AUTH_FILE)

api_key = config["openai"]["key"]
openai.api_key = api_key


@blp.route("/")
class Welcome(MethodView):
    def get(self):
        return {"message": "Welcome. Be sure to review the Swagger docs: /swagger-ui"}


@blp.route("/art/v1/<string:subject>/<string:style>")
class Section(MethodView):
    def get(self, subject, style):
        while True:
            prompt = f"Generate a {subject} in a {style} style using purely HTML and CSS."
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "system", "content": "You are and expert on CSS art "
                                                  "and all response must contain valid HTML code."}
                ]
            )
            content = response.choices[0].message.content.split("```")
            try:
                return content[1]
            except IndexError:
                pass
            