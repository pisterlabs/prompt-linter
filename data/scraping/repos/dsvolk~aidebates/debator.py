from __future__ import annotations

import logging
import string
from typing import Final, List, Optional, Tuple

import pandas as pd

# from langchain.base_language import BaseLanguageModel
# from langchain.chains.base import Chain
from langchain.callbacks import get_openai_callback
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate

from src.chains.basic import BasicChain

# from langchain.callbacks.manager import AsyncCallbackManagerForChainRun, CallbackManagerForChainRun


smart_translator_role_prompts = (
    (
        "idiomatic_translation",
        """Translate this sentence into Russian.

Q: {input}
A:""",
    ),
    (
        "words_translation",
        """List all the words and collocation of intermediate and advanced levels and translate them into Russian. Do not translate simple words. Do not include the translation of the original text.

Example:
Q: Human Self and its nature is held as fundamentally unqualified, faultless, beautiful, blissful, ethical, compassionate and good.
A:
human - человеческий
Self - Я
is held as - считается
unqualified - неквалифицированный
faultless - безупречный
beautiful - красивый
blissful - блаженный
ethical - этичный
compassionate - сострадательный

Q: {input}
A:""",
    ),
    (
        "synonyms",
        """You act as an English tutor who teaches English to a Russian speaker. Your student, whose level is beginner to intermediate, asks you to look at the text and find difficult words phrasal verbs and provide simpler synonyms for them. Don't provide synonyms for simple or intermediate words and do not provide synonyms which are not simple. Only provide very close synonyms. Do not provide synonyms for specialized terms, proper names and numerals. Do not provide synonyms for collocation.

Example:
Q: The conference was abruptly called off due to unforeseen circumstances that occurred at the last minute.
A:
call off - synonym: cancel
occur - synonym: happen

Q: {input}
A:""",
    ),
    (
        "transcription",
        """In the given text choose 20% of the words of the intermediate and advanced level where with the most irregular spelling, along with their transcription.

Example:
Q: Salvation theory occupies a place of special significance in many religions.In the academic field of religious studies, soteriology is understood by scholars as representing a key theme in a number of different religions and is often studied in a comparative context; that is, comparing various ideas about what salvation is and how it is obtained.
A:
soteriology - [soʊtəriˈɑlədʒi]
occupy - [ˈɒkjʊpaɪ]
significance - [sɪɡˈnɪfɪkəns]

Q: {input}
A:""",
    ),
)


smart_translator_role_llm_params: Final[dict] = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.0,
}


class DebatorRole(object):
    def __init__(
        self,
        role_key: str = "smart-translator",
        prompts: Tuple[
            Tuple[str, str], Tuple[str, str], Tuple[str, str], Tuple[str, str]
        ] = smart_translator_role_prompts,
        role_name: str = "Smart Translator",
        pretty_name: str = "🌐 Smart Translator",
        commands: List[str] = None,
        description: str = "",
        hidden: bool = False,
        chains: Optional[Tuple[BasicChain, BasicChain, BasicChain, BasicChain]] = None,
    ):
        self.role_key = role_key
        self.prompts = prompts
        self.role_name = role_name
        self.pretty_name = pretty_name
        self.commands = commands or ["st", "smarttranslator"]
        self.description = description
        self.hidden = hidden
        if chains is None:
            self.chains = (
                BasicChain(
                    prompt=PromptTemplate.from_template(smart_translator_role_prompts[0][1]),
                    llm=ChatOpenAI(**smart_translator_role_llm_params),
                ),
                BasicChain(
                    prompt=PromptTemplate.from_template(smart_translator_role_prompts[1][1]),
                    llm=ChatOpenAI(**smart_translator_role_llm_params),
                ),
                BasicChain(
                    prompt=PromptTemplate.from_template(smart_translator_role_prompts[2][1]),
                    llm=ChatOpenAI(**smart_translator_role_llm_params),
                ),
                BasicChain(
                    prompt=PromptTemplate.from_template(smart_translator_role_prompts[3][1]),
                    llm=ChatOpenAI(**smart_translator_role_llm_params),
                ),
            )
        else:
            self.chains = chains

    def __str__(self):
        return self.role_key

    @classmethod
    def _parse_smart_translator_responses(cls, responses: List[str]) -> str:
        """Parse a protocol string from NVC and return a string with the message."""

        # from pprint import pprint
        # pprint(responses)

        output = responses[0]

        # group 1: words
        words = None
        try:
            words = pd.DataFrame(
                [
                    [line.split(" - ")[0].strip(string.punctuation), line.split(" - ")[1].strip(string.punctuation)]
                    for line in responses[1].strip(string.punctuation).split("\n")
                ],
                columns=["word", "translation"],
            )
        except Exception as e:
            logging.warning(f"Failed to parse words: {e}")

        # group 2: synonyms
        synonyms = None
        try:
            synonyms = pd.DataFrame(
                [
                    [
                        line.split(" - synonym: ")[0].strip(string.punctuation),
                        line.split(" - synonym: ")[1].strip(string.punctuation),
                    ]
                    for line in responses[2].strip(string.punctuation).split("\n")
                ],
                columns=["word", "synonym"],
            )
        except Exception as e:
            logging.warning(f"Failed to parse synonyms: {e}")

        # group 3: transcription
        transcription = None
        try:
            transcription = pd.DataFrame(
                [
                    [line.split(" - ")[0].strip(string.punctuation), line.split(" - ")[1].strip(string.punctuation)]
                    for line in responses[3].strip(string.punctuation).split("\n")
                ],
                columns=["word", "transcription"],
            )
        except Exception as e:
            logging.warning(f"Failed to parse transcription: {e}")

        if words is None:
            return output.strip()
        else:
            # merge all dataframes on word
            df = words
            for df_to_merge in [synonyms, transcription]:
                if df_to_merge is not None:
                    df = df.merge(df_to_merge, on="word", how="outer")

        df = df.dropna(subset=["word", "translation"])
        df = df.drop_duplicates()

        def row_to_text(row):
            text = row["word"]
            if ("transcription" in row) and (pd.notnull(row["transcription"])):
                text += f" - {row['transcription']}"
            if ("translation" in row) and (pd.notnull(row["translation"])):
                text += f" - {row['translation']}"
            if ("synonym" in row) and (pd.notnull(row["synonym"])):
                text += f" (синоним: {row['synonym']})"
            return text

        df["text"] = df.apply(row_to_text, axis=1)

        output += "\n\n" + "\n".join(df["text"].tolist())

        return output.strip()

    async def predict(self, text: str):
        with get_openai_callback() as cb:
            responses = [
                await chain.arun({"input": text}, callbacks=[StdOutCallbackHandler()]) for chain in self.chains
            ]

        logging.info(f"LLM response: {responses}")

        try:
            parsed_response = self._parse_smart_translator_responses(responses)
        except Exception as e:
            logging.warning(f"Failed to parse LLM response: {e}")
            parsed_response = "\n\n".join(responses)

        return {
            "text": parsed_response,
            "n_prompt_tokens": cb.prompt_tokens,
            "n_completion_tokens": cb.completion_tokens,
        }
