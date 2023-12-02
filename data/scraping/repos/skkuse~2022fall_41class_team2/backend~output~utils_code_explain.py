from openai import Completion

DESCRIPTION_SUPPORT_LANGUAGE = ['python']

ENGINE = 'davinci'
STOP = ["\"\"\""]
TOP_P = 1.0
BEST_OF = 1
MAX_TOKENS = 64
TEMPERATURE = 0


def run(raw_code: str, language: str = 'python'):
    if language.lower() not in DESCRIPTION_SUPPORT_LANGUAGE:
        return str()

    return execute_codex(raw_code)


def execute_codex(raw_code: str):
    prompt = raw_code + "\n\"\"\"\nCode Explain:\n1."

    response = Completion.create(
        prompt=prompt,
        engine=ENGINE,
        stop=STOP,
        top_p=TOP_P,
        best_of=BEST_OF,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    answer = response.choices[0].text.strip()
    return "1. " + answer
