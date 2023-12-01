from dataclasses import dataclass, asdict
import openai

@dataclass
class OpenAI:
    model: str = "davinci"
    # suffix: Optional[str] = None
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    # logprobs: Optional[int] = None
    echo: bool = False
    # stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    best_of: int = 1
    # logit_bias: Optional[Dict[str, float]] = None
    # user: Optional[str] = None

    def complete(self, prompt: str, **kwargs) -> openai.Completion:
        return openai.Completion.create(prompt=prompt, **asdict(self), **kwargs)

def make_model(prompt, llm=OpenAI(temperature=0), callbacks=[]):
    def func(*args, **kwargs):
        inputs = prompt.format(*args, **kwargs)
        completion = llm.complete(inputs, stop="\n")
        model_metadata = asdict(llm)
        model_metadata.update({"prompt": prompt})
        model_id=hash(frozenset(model_metadata.items()))
        for callback in callbacks:
            callback(prompt=prompt, args=args, kwargs=kwargs, completion=completion, llm=llm, model_id=model_id)
        return completion.choices[0].text
    return func
