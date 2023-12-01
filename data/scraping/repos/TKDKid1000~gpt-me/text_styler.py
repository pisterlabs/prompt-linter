import os
import re
from collections import defaultdict
from typing import Generic, TypedDict, TypeVar

import cohere
import pandas as pd
from nltk.corpus import words as nltk_words
from nltk.tokenize import sent_tokenize, word_tokenize
import random
from tqdm import tqdm
from transformers import Pipeline, pipeline
from typing_extensions import NotRequired

from gptme.utils.regex import trim_lines

contractions_df = pd.read_csv("data/contractions.csv")
abbreviations_df = pd.read_csv("data/contractions.csv")


T = TypeVar("T")


class SentenceTypes(TypedDict, Generic[T]):
    declarative: NotRequired[T]
    exclamation: NotRequired[T]
    question: NotRequired[T]


class Punctuation(TypedDict):
    sentence_endings: SentenceTypes[dict[str, float]]
    proper_capitalization: SentenceTypes[float]
    types: SentenceTypes[int]


class Contractions(TypedDict):
    total: int
    proper: float
    improper: float
    expanded: float


class Diction(TypedDict):
    formal: float
    informal: float
    pedantic: float
    pedestrian: float
    slang: float
    colloquial: float
    abstract: float
    concrete: float
    poetic: float


class AverageLengths(TypedDict):
    words: int
    sentences: int


class Style(TypedDict):
    punctuation: Punctuation
    contractions: Contractions
    diction: Diction
    lengths: AverageLengths
    emojis: float
    mood: list[str]

    # exclamation_points: bool
    # question_marks: bool
    # commas: bool
    # proper_capitalization: bool
    # formality: Literal["casual", "neutral", "polite", "formal", "professional"]
    # grammar: Literal["novice", "beginner", "intermediate", "competent", "advanced"]
    # vocabulary: Literal["basic", "intermediate", "competent", "proficient", "advanced"]
    # contractions: Literal["always", "often", "sometimes", "seldom", "never"]
    # ellipsis: Literal["always", "often", "sometimes", "seldom", "never"]
    # abbreviations: Literal["always", "often", "sometimes", "seldom", "never"]
    # emojis: Literal["always", "often", "sometimes", "seldom", "never"]
    # style: list[str]
    # diction: list[str]
    # mood: list[str]


class NoStyleProvidedError(Exception):
    pass


class TextStyler:
    style: Style
    cohere_client: cohere.Client
    roberta_large_mnli: Pipeline

    def __init__(self) -> None:
        self.cohere_client = cohere.Client(os.environ["COHERE_KEY"])
        self.roberta_large_mnli = pipeline(
            "zero-shot-classification", model="roberta-large-mnli"
        )

    def extract_style(self, samples: list[str]):
        lengths = self._average_lengths(samples)
        diction = self._classify_diction(samples)
        punctuation = self._calculate_punctuation_percentages(samples)
        contractions = self._count_contractions(" ".join(samples))
        emojis = self._calculate_emoji_percentage(samples)
        mood = self._generate_cohere_mood("\n".join(samples))
        print(mood)

        self.style = {
            "punctuation": punctuation,
            "contractions": contractions,
            "diction": diction,
            "lengths": lengths,
            "emojis": emojis,
            "mood": mood,
        }

    def _average_lengths(self, samples: list[str]) -> AverageLengths:
        words: list[str] = sum([word_tokenize(sample) for sample in samples], [])
        sentences: list[str] = sum([sent_tokenize(sample) for sample in samples], [])

        return {
            "words": round(len(words) / len(samples)),
            "sentences": round(len(sentences) / len(samples)),
        }

    def _classify_diction(self, samples: list[str]) -> Diction:
        results = {
            "formal": 0,
            "informal": 0,
            "pedantic": 0,
            "pedestrian": 0,
            "slang": 0,
            "colloquial": 0,
            "abstract": 0,
            "concrete": 0,
            "poetic": 0,
        }

        for sample in tqdm(samples, desc="Diction"):
            sample_diction: str = self.roberta_large_mnli(
                sample,
                [
                    "formal",
                    "informal",
                    "pedantic",
                    "pedestrian",
                    "slang",
                    "colloquial",
                    "abstract",
                    "concrete",
                    "poetic",
                ],
            )[  # type: ignore
                "labels"
            ][
                0
            ]
            results[sample_diction.split(" ")[0]] += 1

        return {diction: count / len(samples) for diction, count in results.items()}  # type: ignore

    def _count_contractions(self, sentence: str) -> Contractions:
        sentence = re.sub(r"[‘’]", "'", sentence)  # Normalize single quotations.
        sentence = re.sub(r"[^\sa-z0-9']", "", sentence.lower())
        proper = 0
        improper = 0
        expanded = 0

        for row in contractions_df.itertuples():
            found = sentence.count(row.meaning)
            expanded += found

        words = sentence.lower().split(" ")
        for word in tqdm(words, desc="Contractions"):
            for row in contractions_df.itertuples():
                if word == row.contraction:
                    proper += 1
                elif (
                    word == row.contraction.replace("'", "")
                    and word not in nltk_words.words()
                ):
                    improper += 1
        total = proper + improper + expanded

        return {
            "total": total,
            "proper": proper / total,
            "improper": improper / total,
            "expanded": expanded / total,
        }

    def _calculate_emoji_percentage(self, samples: list[str]):
        words: list[str] = sum([word_tokenize(sample) for sample in samples], [])
        total_words = len(words)
        emojis = 0

        for word in words:
            emojis += len(
                re.findall(
                    # pylint: disable=line-too-long
                    r"[\U0001F300-\U0001F64F\U0001F680-\U0001F6FF\U00026000-\U00026FFF\U00002700-\U000027BF\U0001F900-\U0001F9FF\U0001F1E6-\U0001F1FF]+",
                    word,
                )
            )

        emojis_percentage = emojis / total_words

        return emojis_percentage

    def _calculate_punctuation_percentages(self, samples: list[str]) -> Punctuation:
        sentencetypes_int: SentenceTypes[int] = {
            "declarative": 0,
            "exclamation": 0,
            "question": 0,
        }

        proper_capitalization: SentenceTypes[int] = {**sentencetypes_int}
        ending_punctuation: SentenceTypes[dict[str, int]] = {
            "declarative": defaultdict(int),
            "exclamation": defaultdict(int),
            "question": defaultdict(int),
        }
        commas: dict[str, int] = {**sentencetypes_int}
        interior_ellipsis: SentenceTypes[int] = {**sentencetypes_int}
        sentence_types: SentenceTypes[int] = {**sentencetypes_int}

        sentences: list[str] = sum([sent_tokenize(sample) for sample in samples], [])
        total_sentences = len(sentences)

        normalizations = {
            r"[“”„”«»]": '"',  # Double quotations.
            r"[‘’]": "'",  # Single quotations.
            r"…": "...",  # Ellipsis
        }

        for sentence in tqdm(sentences, desc="Sentences"):
            sentence_type: str = self.roberta_large_mnli(
                sentence, ["declarative", "exclamation", "question"]
            )[  # type: ignore
                "labels"
            ][
                0
            ]
            sentence_types[sentence_type] += 1
            for expression, replacement in normalizations.items():
                sentence = re.sub(expression, replacement, sentence)
            if len(sentence) < 2:
                # Short corrections or errors can be ignored.
                total_sentences -= 1
                continue
            if re.search(r"(?:^[^a-z\"(][^A-Z])|(?:[\"(])[A-Z]", sentence):
                # Matches `Ab`, `"A`, or `(A` sentences. A is uppercase, b is lowercase.
                proper_capitalization[sentence_type] += 1
            special_tail = re.search(r"[^a-zA-Z0-9\s]+$", sentence)
            if special_tail:
                ending_punctuation[sentence_type][special_tail.group()] += 1
            else:
                ending_punctuation[sentence_type][""] += 1

            interior_ellipsis[sentence_type] += sentence.rstrip(".").count("...")
            commas[sentence_type] += sentence.count(",")

        ending_punctuation_percentages: SentenceTypes[dict[str, float]] = {
            sentence_type: {
                punctuation: count / sentence_types[sentence_type]
                for punctuation, count in endings.items()  # type: ignore
            }
            for sentence_type, endings in ending_punctuation.items()
        }

        proper_capitalization_percentages: SentenceTypes[float] = {
            sentence_type: count / sentence_types[sentence_type]
            for sentence_type, count in proper_capitalization.items()  # type: ignore
        }

        return {
            "sentence_endings": ending_punctuation_percentages,
            "proper_capitalization": proper_capitalization_percentages,
            "types": sentence_types,
            # "interior_ellipsis": interior_ellipsis / total_sentences,
            # "commas": commas / total_sentences,
        }

    def _generate_cohere_mood(self, samples: str) -> list[str]:
        response = self.cohere_client.generate(
            model="command-nightly",
            prompt=trim_lines(
                f"""Describe the mood of the sample text using the five bullet point template provided.
                Each point should contain one or two words.

                Example response:
                1. adjective
                2. adjective
                3. adjective
                4. adjective
                5. adjective

                Samples:

                {samples}
                """
            ),
            temperature=0.7,
            return_likelihoods="NONE",
        )
        if response.generations is None:
            raise RuntimeError()

        response_text: str = response.generations[0].text
        print(response_text.strip())

        regex = r"^\d. (.+)"
        matches = re.findall(regex, response_text.lower().strip(), re.MULTILINE)

        return matches

    def _category_to_str(self, category: str | bool | list[str]):
        if isinstance(category, str):
            return category
        if isinstance(category, bool):
            return "yes" if category else "no"
        if isinstance(category, list):
            return ", ".join(category)
        return ""

    def apply_style(self, text: str):
        if self.style is None:
            raise NoStyleProvidedError("Style not found.")

        diction = random.choices(
            population=[*self.style["diction"]],
            weights=[*self.style["diction"].values()],  # type: ignore
            k=1
        )[0]
        lengths = self.style["lengths"]

        cohere_rewrite = self.cohere_client.generate(
            model="command-nightly",
            prompt=trim_lines(
                f"""Rewrite the following sentence in a {diction} diction.
                Use a {self.style["mood"][:-1]} and {self.style["mood"][-1]} mood.
                Use roughly {lengths["sentences"]} sentences or {lengths["words"]} words.

                {text}
                """
            ),
            max_tokens=self.style["lengths"]["words"] * 4,
            temperature=0.7,
        )
        if cohere_rewrite.generations is None:
            raise RuntimeError()
        
        rewrite_text: str = cohere_rewrite.generations[0].text

        
