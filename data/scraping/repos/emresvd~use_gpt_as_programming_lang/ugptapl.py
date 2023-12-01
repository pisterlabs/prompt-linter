import openai
import dotenv
import os


class UGPTapl(object):
    def __init__(self, code: str) -> None:
        super().__init__()

        dotenv.load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")

        self.prompt = code + \
            "\ntranslate the above algorithm to python code and write only python code"
        self.python_code: str = None

    def translate_to_python_code(self) -> None:
        completion = openai.Completion.create(
            engine="text-davinci-003",
            prompt=self.prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        self.python_code = completion.choices[0].text

    def run(self) -> None:
        exec(self.python_code)
