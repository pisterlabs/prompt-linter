import os
from re import sub
import requests
import openai

from prompts import positive_prompt, negative_prompt


class ArticleGenerator:
    def __init__(self):
        self.api_key = None
        self.model_choices = []

    def raise_wrong_model(self):
        raise Exception(
            f"Model not supported, please use one of the following: {self.model_choices}"
        )

    def generate_completion(self, *args, **kwargs):
        raise NotImplementedError(
            f"generate_completion is not implemented for the class {self.__class__.__name__}"
        )

    def generate_title(self, model, subject):
        # TODO
        raise NotImplementedError()

    def generate_abstract(self, model, subject):
        # TODO
        raise NotImplementedError()

    def generate_conclusion(self, model, subject):
        # TODO
        raise NotImplementedError()

    def generate_article(self, model, subject, refute):
        # TODO: For a complete article, generate multiple "texts" with multiple queries: "title", "abstract", "core" and "conclusion"
        # title = self.generate_title(model, subject)
        # abstract = self.generate_abstract(model, subject)8
        prompt = ""
        if refute:
            prompt = negative_prompt.format(subject=subject)
        else:
            prompt = positive_prompt.format(subject=subject)

        params = {
            "num_results": 1,
            "max_tokens": 450,
            "stopSequences": ["#####", "Subject: "],
            "temperature": 0.8,
            "topKReturn": 2,
            # topP = 1.0
            "frequency_penalty": 0.3,
            "presence_penalty": 0.3,
        }

        core = self.generate_completion(model, prompt, params)

        return "\n\n".join([core])


class ArticleGeneratorOpenAI(ArticleGenerator):
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        openai.api_key = self.api_key

        self.model_choices = [e["id"] for e in self.list_engines()]

    def list_engines(self):
        return openai.Engine.list()["data"]

    def generate_completion(self, model, prompt, params):
        if not model in self.model_choices:
            self.raise_wrong_model()

        if params.get("stopSequences", None) == []:
            params["stopSequences"] = None

        res = openai.Completion.create(
            engine=model,
            prompt=prompt,
            temperature=params.get("temperature", 0.8),
            max_tokens=params.get("maxTokens", 200),
            # top_p=params.get("topP", 1.0),
            frequency_penalty=params.get("frequency_penalty", 0.3),
            presence_penalty=params.get("presence_penalty", 0.3),
            stop=params.get("stopSequences", None),
        )

        if res != 0:
            for choice in res.choices:
                return choice.text.strip()
        else:
            raise Exception("No response found")


class ArticleGeneratorAI21(ArticleGenerator):
    def __init__(self):
        self.api_key = os.environ.get("AI21_API_KEY")
        self.model_choices = ["j1-large", "j1-jumbo"]

    def get_api_url(self, model):
        return f"https://api.ai21.com/studio/v1/{model}/complete"

    def generate_completion(self, model, prompt, params):
        if not model in self.model_choices:
            self.raise_wrong_model()

        if params.get("stopSequences", None) is None:
            params["stopSequences"] = []

        res = requests.post(
            self.get_api_url(model),
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "prompt": prompt,
                "numResults": params.get("numResults", 1),
                "maxTokens": params.get("maxTokens", 200),
                "stopSequences": params.get("stopSequences", None),
                "topKReturn": params.get("topKReturn", 2),
                "temperature": params.get("temperature", 0.8),
                # "topP": params.get("topP", 1.0),
            },
        )

        if res.status_code != 200:
            raise Exception(
                f"Failed request on {self.get_api_url(model)}. {res.json()}"
            )

        completions = [c["data"]["text"] for c in res.json()["completions"]]
        return "\n".join(completions).strip("\n").strip()
