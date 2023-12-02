import json
import os

import cohere

from lib.handler import DiscordAPI, InteractionCallbackType


class YourMumDiscordAPI(DiscordAPI):
    COMMAND_NAME = "mum_joke"

    def __init__(self):
        super().__init__()

        with open("config.json", "r") as f:
            config = json.load(f)

        template = []
        template.append(config["introduction"])
        template.append("")
        for sample in config["samples"]:
            template += [
                "Prompt: " + sample["prompt"],
                "Your mum joke: " + sample["joke"],
                "--",
            ]
        template += [
            "Prompt: {prompt}",
            "Your mum joke:",
        ]

        self.template = "\n".join(template)
        self.co = cohere.Client(os.environ["COHERE_API_KEY"])
        self.default = config["defaultResponse"]

    def make_joke(self, prompt: str):
        try:
            prediction = self.co.generate(
                model="xlarge",
                prompt=self.template.format(prompt=prompt),
                temperature=0.4,
                k=0,
                p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop_sequences=["--", "\n"],
                num_generations=1,
            )

            joke = prediction.generations[0].text
            if "your mum" not in joke.lower():
                return None
            return joke
        except cohere.CohereError:
            return None

    def custom_handler(self, body):
        data = body["data"]
        command = data["name"]
        interaction_id = body["id"]
        interaction_token = body["token"]

        if command == self.COMMAND_NAME:
            print(f"Received {command}.")

            prompts = [
                option for option in data["options"] if option["name"] == "prompt"
            ]
            assert len(prompts) == 1

            # shows "<Bot Name> is thinking..."
            response = self.create_interaction_response(
                interaction_id,
                interaction_token,
                {"type": InteractionCallbackType.DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE},
            )
            if not response.ok:
                print(response.json())

            prompt = prompts[0]["value"]
            joke = self.make_joke(prompt)

            if joke is None:
                content = self.default
            else:
                content = f"> *{prompt}*\n{joke}"

            # sends the response through http API
            response = self.create_followup_message(
                interaction_token, {"content": content}
            )
            if not response.ok:
                print(response.json())

            return {}

        print(f"Unknown command: {command}")
        return self.make_response(400, {"msg": "unhandled command"})


discord_api = YourMumDiscordAPI()


def main(args: dict):
    try:
        return discord_api.handle(event=args["http"])
    except Exception as e:
        print(e)
        return YourMumDiscordAPI.make_response(
            500, {"msg": "An interval server error occurred. Please view the logs."}
        )
