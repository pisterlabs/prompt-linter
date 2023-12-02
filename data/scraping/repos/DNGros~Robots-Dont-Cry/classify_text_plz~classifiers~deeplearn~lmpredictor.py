import random
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple, Any, List, Union

from joblib import Memory
from termcolor import colored
from openai.openai_object import OpenAIObject
cur_file = Path(__file__).parent.absolute()
import openai


@dataclass(frozen=True)
class LmPrompt:
    text: str
    max_toks: int
    stop: List[str] = None
    logprobs: int = None
    temperature: float = 1.0
    top_p: float = 0.9,
    presence_penalty: float = 0.0

@dataclass
class LmPrediction:
    text: str
    metad: Any


class LmPredictor:
    @abstractmethod
    def predict(
        self,
        prompt: Union[str, LmPrompt],
    ) -> LmPrediction:
        pass

    def _cast_prompt(self, prompt: Union[str, LmPrompt]) -> LmPrompt:
        if isinstance(prompt, str):
            return LmPrompt(prompt, 100)
        return prompt

    def model_name(self):
        return self.__class__.__name__


cachedir = 'autoregressive_prompt_model_cache'
diskcache = Memory(cachedir, verbose=0)


class OpenAIPredictor(LmPredictor):
    def __init__(
        self,
        api,
        engine_name: str,
        cache_outputs: bool = True,
    ):
        if not cache_outputs:
            raise NotImplementedError
        self._api = api
        self._engine_name = engine_name
        self._cache_outputs = cache_outputs

    def model_name(self):
        return self._engine_name

    def predict(
        self,
        prompt: Union[str, LmPrompt],
    ) -> LmPrediction:
        prompt = self._cast_prompt(prompt)
        return _openai_cache_prediction(self._api, self._engine_name, prompt)


@diskcache.cache(ignore=['api'], verbose=0)
def _openai_cache_prediction(
    api,
    engine_name,
    prompt,
):
    print("PREDICT", prompt)
    completion = api.Completion.create(
        engine=engine_name,
        prompt=prompt.text,
        max_tokens=prompt.max_toks,
        stop=prompt.stop,
        stream=False,
        logprobs=prompt.logprobs,
        temperature=prompt.temperature,
        top_p=prompt.top_p,
        presence_penalty=prompt.presence_penalty,
    )
    if random.random() < 0.05:
        print("PREDICT", "\n" + colored(prompt.text, 'blue'))
        print("GOT", colored(completion.choices[0].text, 'green'))
    return LmPrediction(completion.choices[0].text, completion)




class FoobarPredictor(LmPredictor):
    def __init__(
        self,
    ):
        pass

    def predict(
        self,
        prompt: Union[str, LmPrompt],
    ) -> LmPrediction:
        prompt = self._cast_prompt(prompt)
        print("PREDICT", colored(prompt.text, 'blue'))
        return LmPrediction("Foobar", None)


def get_goose_lm(
    model_name: str = "gpt-neo-125m",
):
    import openai
    openai.api_key = Path("~/goose_key.txt").expanduser().read_text().strip()
    openai.api_base = "https://api.goose.ai/v1"
    return OpenAIPredictor(
        api=openai,
        engine_name=model_name,
    )


def get_open_ai_lm(
    model_name: str = "gpt-neo-125m",
):
    import openai
    openai.api_key = Path("~/oai_key.txt").expanduser().read_text().strip()
    return OpenAIPredictor(
        api=openai,
        engine_name=model_name,
    )
# List Engines (Models)
    #engines = openai.Engine.list()
    ## Print all engines IDs
    #for engine in engines.data:
    #    print(engine.id)

    #print(engines)

    #return OpenAIPredictor("gpt-j-6b")

    # Create a completion, return results streaming as they are generated. Run with `python3 -u` to ensure unbuffered output.
    #completion = openai.Completion.create(
    #    engine="gpt-j-6b",
    #    prompt="Once upon a time there was a Goose. ",
    #    max_tokens=160,
    #    stream=False
    #)

    ## Print each token as it is returned
    #for c in completion:
    #    print(c.choices[0].text, end='')

    #print("")


def main():
    lm = get_goose_lm()
    text = lm.predict(
        LmPrompt(
            "Once upon a time there was a Goose. And",
            max_toks=1, logprobs=10,
        ))
    #print(type(text))
    print(text.text)
    text = lm.predict(
        LmPrompt(
            "Once upon a time there was a Goose. And",
            max_toks=1, logprobs=10,
        ))
    #print(type(text))
    print(text.text)


if __name__ == "__main__":
    main()