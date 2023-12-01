import json
from openai import Completion
import openai
import requests
import sseclient
from .base_model import HTTPAPILLMModel


class TigerAPILLMModel(HTTPAPILLMModel):
    def __init__(self, api_key, api_base, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_key = api_key
        self.api_base = api_base
        self.model = None

    def response(
        self,
        prompt,
        num_trials=3,
        **kwargs,
    ):
        success, trials, errors = False, 0, []

        while (not success) and (trials < num_trials):
            try:
                session = requests.Session()
                data = {"prompt": prompt, "history": []}
                headers = {"Content-Type": "application/json"}
                event_source = sseclient.SSEClient(
                    self.api_base, json=data, headers=headers, session=session
                )
                history = []
                for event in event_source:
                    # 将事件传递给回调函数进行处理
                    data = json.loads(event.data)
                    if not data["finish"]:
                        response = data["response"]
                        history = data["history"]
                        print("\rChatBot:" + response, end="")
                    else:
                        break

                generated_text = Completion["choices"][0]["text"]
                success = True
            except Exception as ex:
                generated_text = f"\n[Error] {type(ex).__name__}: {ex}\n"
                errors.append(generated_text)
            finally:
                trials += 1

        if not success:
            generated_text = "".join(errors)
            print(errors[-1])
        return {
            "success": success,
            "trails": trials,
            "prompt": prompt,
            "messages": [],
            "generated_text": generated_text,
        }

    def chat_response(
        self,
        messages,
        num_trials=3,
        **kwargs,
    ):
        success, trials, errors = False, 0, []

        while (not success) and (trials < num_trials):
            try:
                chat_completion = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    echo=False,
                    n=1,
                    stream=False,
                    **kwargs,
                )
                generated_text = chat_completion["choices"][0]["message"]["content"]
                success = True
            except Exception as ex:
                generated_text = f"\n[Error] {type(ex).__name__}: {ex}\n"
                errors.append(generated_text)
            finally:
                trials += 1

        if not success:
            generated_text = "".join(errors)
            print(errors[-1])
        return {
            "success": success,
            "trails": trials,
            "prompt": None,
            "messages": messages,
            "generated_text": generated_text,
        }
