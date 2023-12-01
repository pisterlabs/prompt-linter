import openai
import os
import anthropic

class AI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        # try:
        #     openai.Model.retrieve("gpt-4")
        # except openai.error.InvalidRequestError:
        #     print("Model gpt-4 not available for provided api key reverting "
        #           "to gpt-3.5.turbo. Sign up for the gpt-4 wait list here: "
        #           "https://openai.com/waitlist/gpt-4-api")
        #     self.kwargs['model'] = "gpt-3.5-turbo"

    def start(self, system, user):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        return self.next(messages)

    def fsystem(self, msg):
        return {"role": "system", "content": msg}

    def fuser(self, msg):
        return {"role": "user", "content": msg}

    def next(self, messages: list[dict[str, str]], prompt=None):
        chat = []

        if prompt:
            messages = messages + [{"role": "user", "content": prompt}]
        if "gpt" in self.kwargs['model']:
            response = openai.ChatCompletion.create(
                messages=messages, stream=True, **self.kwargs
            )
            for chunk in response:
                delta = chunk["choices"][0]["delta"]
                msg = delta.get("content", "")
                print(msg, end="")
                chat.append(msg)
        elif "claude" in self.kwargs['model']:
            c = anthropic.Client(os.environ["ANTHROPIC_API_KEY"])

            response = c.completion(
                prompt=f"{anthropic.HUMAN_PROMPT}Previous Messages\n{messages}/n{anthropic.AI_PROMPT}",
                stop_sequences=[],
                model=self.kwargs['model'],
                max_tokens_to_sample=10000)
            chat.append(response['completion'])
            

        return messages + [{"role": "assistant", "content": "".join(chat)}]
