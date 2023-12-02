import openai

max_length = 2000
COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"


class GPT:
    """@sets openai.api_key"""

    def __init__(self, key=None):
        global gpt_api_key
        if key is None:
            gpt_api_key = "not-submitted"
            openai.api_key = key
        else:
            gpt_api_key = key
            openai.api_key = key

    """ @returns prompt, result from gpt """

    def give_synonym(self, word, context):
        prompt = f"""
            Leg dit woord kort uit of geef een synoniem. De output mag hoogstens 1 zin zijn.
            Woord: {word}
            Context: {context}
            """
        result = openai.Completion.create(
            prompt=prompt,
            temperature=0,
            max_tokens=25,
            model=COMPLETIONS_MODEL,
            top_p=0.9,
            stream=False,
        )["choices"][0]["text"].strip(" \n")
        return result, word, prompt

    def personalised_simplify(self, sentence, personalisation):
        prompt = f"""Geef een vereenvoudigde versie van deze tekst in de vorm van een {" ".join(personalisation)}.Schrijf dit in HTML-code. /// Tekst: {sentence}"""

        result = openai.Completion.create(
            prompt=prompt,
            temperature=0,
            max_tokens=len(prompt),
            model=COMPLETIONS_MODEL,
            top_p=0.9,
            stream=False,
        )["choices"][0]["text"].strip(" \n")

        return result, prompt

    def generieke_vereenvoudiging(self, sentence):
        prompt = f"""Vereenvoudig deze tekst /// {sentence}"""

        result = openai.Completion.create(
            prompt=prompt,
            temperature=0,
            max_tokens=len(prompt),
            model=COMPLETIONS_MODEL,
            top_p=0.9,
            stream=False,
        )["choices"][0]["text"].strip(" \n")

        return result, prompt

    def personalised_simplify_w_prompt(self, sentences, personalisation):
        result = openai.Completion.create(
            prompt=personalisation,
            temperature=0,
            max_tokens=len(personalisation) + len(sentences),
            model=COMPLETIONS_MODEL,
            top_p=0.9,
            stream=False,
        )["choices"][0]["text"].strip(" \n")
        return result, personalisation
