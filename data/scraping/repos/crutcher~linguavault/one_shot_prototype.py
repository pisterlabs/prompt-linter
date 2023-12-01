import argparse
import sys
from typing import Optional

import openai
from marshmallow_dataclass import dataclass

from linguavault.api_utils import load_openai_secrets

DEFINE_PREFIX = """
You are an opinionated dictionary of all the world's languages.

Your definitions should be accurate to common usage, but also informal and cheeky, humorous and anti-authoritarian.

You should write in the style of the Hitchhiker's Guide to the Galaxy.

Word-play is encouraged, provided it doesn't obscure the meanings of the terms.

When prompted, you should include all the well-known senses (or meanings) of the term, ordered by their popularity.

Include professional, popular, urban, and technical senses for the words.
Include newer popular slang and idiom senses.
Include subversive and oppositional critique of the sense in the definition.

The user will provide a prompt; and you should respond with a well-formed structured JSON object.

The term language (the language the term is defined in) will be provided by the user.
The desired output language for the results to be written in will be provided by the user.
The term itself will be provided by the user.

An optional term context may be provided to specify the context in which the term is found.

Example 1:
TermLanguage: <TermLanguage>
OutputLanguage: <OutputLanguage>
Term: <Term>

Example 2:
TermLanguage: <TermLanguage>
OutputLanguage: <OutputLanguage>
Term: <Term>
TermContext: <TermContext>

The structure of the language object should be:
{
  "term": <Term>,
  "term_language": <Term Language>,
  "term_context": <TermContext, or null>,
  "output_language": <Output Language>,
  "senses": [
    {
      "part_of_speech": <Part of Speech>,
      "keywords": [<Keyword of contexts where the word-sense might be seen> ...],
      "popularity": <Degree of Popularity as a float from 0.0 to 1.0>,
      "formality": <Degree of Formality as a float from 0.0 to 1.0>,
      "vulgarity": <Degree of Vulgarity as a float from 0.0 to 1.0>,
      "synonyms": [<Synonym> ...],
      "antonyms": [<Antonym; term that means the opposite of the sense> ...],
      "short_definition": <Short Definition written in the given Output Language>,
      "long_definition": <Long Definition (~500 words) written in the given Output Language>,
      "example_usage": <Example Sentence written in the given Term Language>
    }
    < Additional Senses ... >
  ]
}

Try hard to include all well-known senses of a term.
Do not produce any text other than the JSON output.
"""


@dataclass
class TermSenseDefinition:
    part_of_speech: str
    popularity: float
    formality: float
    vulgarity: float
    short_definition: str
    long_definition: str
    example_usage: str
    keywords: Optional[list[str]] = None
    synonyms: Optional[list[str]] = None
    antonyms: Optional[list[str]] = None


@dataclass
class TermDefinition:
    term: str
    term_language: str
    term_context: Optional[str]
    output_language: str
    senses: list[TermSenseDefinition]


def define_term(
    term: str,
    term_language: str = "English",
    term_context: Optional[str] = None,
    output_language: str = "English",
) -> tuple[TermDefinition, str]:
    query = [
        f"TermLanguage: {term_language}",
        f"OutputLanguage: {output_language}",
        f"Define: {term}",
    ]
    if term_context:
        query.append(
            f"TermContext: {term_context}",
        )

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            dict(role="system", content=DEFINE_PREFIX),
            dict(role="user", content="\n".join(query)),
        ],
    )
    answer = completion["choices"][0]["message"]["content"]
    try:
        return TermDefinition.Schema().loads(answer), answer  # type: ignore
    except Exception as e:
        raise ValueError(answer) from e


def display_term(term: TermDefinition) -> None:
    print(
        f'"{term.term}" :: source={term.term_language} :: output={term.output_language}'
    )
    if term.term_context:
        print(f"Context: {term.term_context}")

    for sense in term.senses:
        print()
        print(f"{sense.part_of_speech} : {', '.join(sense.keywords or [])}")
        print(sense.short_definition)
        print(
            f"Usage: popularity={sense.popularity}, formality={sense.formality}, vulgarity={sense.vulgarity}"
        )
        if sense.synonyms:
            print(f"Synonyms: {', '.join(repr(t) for t in sense.synonyms)}")
        if sense.antonyms:
            print(f"Antonyms: {', '.join(repr(t) for t in sense.antonyms)}")
        print(f'Example: "{sense.example_usage}"')
        print()
        print(sense.long_definition)


def query(
    term: str,
    term_language: str = "English",
    term_context: Optional[str] = None,
    output_language: str = "English",
) -> tuple[TermDefinition, str]:
    td, raw = define_term(
        term,
        term_language=term_language,
        term_context=term_context,
        output_language=output_language,
    )
    display_term(td)
    return td, raw


def main(argv):
    p = argparse.ArgumentParser(prog="dictionary")
    p.add_argument("--secrets_file", default=None)
    p.add_argument("term", help="The term to define")
    p.add_argument(
        "--term_context",
        default=None,
        help="The context the term is found in.",
    )
    p.add_argument(
        "--term_language",
        default="English",
        help="The language the term is in.",
    )
    p.add_argument(
        "--output_language",
        default="English",
        help="The language the output should be written in.",
    )
    args = p.parse_args(argv)

    load_openai_secrets(args.secrets_file)

    _td, _raw = query(
        term=args.term,
        term_language=args.term_language,
        term_context=args.term_context,
        output_language=args.output_language,
    )
    print()


if __name__ == "__main__":
    main(sys.argv[1:])
