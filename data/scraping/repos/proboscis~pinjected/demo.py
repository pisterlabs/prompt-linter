from pathlib import Path

from pinjected import Design, instances, injected_function, Injected
import openai

__meta_design__: Design = instances(
    default_design_paths=["pinjected.demo.default_design"]
)

default_design: Design = instances(
    openai_api_key="my secret key",
    model="text-davinci-003",
    max_tokens=1000,
    load_openai_api_key=lambda: Path("~/openai_api_key.txt").expanduser().read_text().strip()
)


@injected_function
def LLM(load_openai_api_key: str, model, max_tokens, /, prompt) -> str:
    # openai_api_key = Path("~/openai_api_key.txt").expanduser().read_text().strip()
    return openai.Completion.create(
        load_openai_api_key(), prompt=prompt, model=model, max_tokens=max_tokens
    )["choices"][0]["text"]


@injected_function
def Gandalf(LLM: "str -> str", /, user_message: str) -> str:
    return LLM(f"Respond to the user's message as if you were a Gandalf: {user_message}")


test_greeting: Injected = Gandalf("How are you?")

