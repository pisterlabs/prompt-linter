#!/usr/bin/env python
"""
Non-Player Character Briefing Sheet
===================================

This Python module uses OpenAI ChatGPT to generate a non-player character
briefing sheet based on the
https://thealexandrian.net/wordpress/37916/roleplaying-games/universal-npc-roleplaying-template
template.

Notes
-----
-   An OpenAI API key is required!
"""

import openai
from dataclasses import dataclass

__copyright__ = "Copyright 2023 Thomas Mansencal"
__license__ = "MIT License - https://opensource.org/licenses/MIT"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = [
    "Summary",
    "format_non_player_character_summary",
    "content_from_response",
    "generate_non_player_character_briefing_sheet",
    "format_non_player_character_briefing_sheet",
]


@dataclass
class Summary:
    name: str
    race: str
    alignment: str
    class_: str | None = None
    context: str | None = None


def format_non_player_character_summary(summary):
    formatted_summary = f"{summary.name} is a {summary.alignment} {summary.race}"
    if summary.class_ is not None:
        formatted_summary += f" {summary.class_}"

    if summary.context is not None:
        formatted_summary += f" {summary.context}"

    if not formatted_summary.endswith("."):
        formatted_summary += "."

    return formatted_summary


def content_from_response(response):
    return response["choices"][0]["message"]["content"]


def generate_non_player_character_briefing_sheet(summary):
    model = "gpt-3.5-turbo"
    formatted_summary = format_non_player_character_summary(summary)

    history = (
        "You are an assistant creating fictional Non-Character Player "
        "briefing sheet for Dungeon & Dragons 5e."
    )
    messages = [
        {
            "role": "system",
            "content": history,
        },
        {"role": "user", "content": formatted_summary},
    ]
    messages += [
        {
            "role": "user",
            "content": f"Describe {summary.name} appearance.",
        }
    ]

    history += formatted_summary

    briefing_sheet = {
        "appearance": content_from_response(
            openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        )
    }
    history += f'\n{briefing_sheet["appearance"]}'

    briefing_sheet["personality"] = content_from_response(
        openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "assistant",
                    "content": history,
                },
                {
                    "role": "user",
                    "content": f"Describe {summary.name} personality.",
                },
            ],
        )
    )
    history += f'\n{briefing_sheet["personality"]}'

    briefing_sheet["background"] = content_from_response(
        openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "assistant",
                    "content": history,
                },
                {
                    "role": "user",
                    "content": f"Describe {summary.name} background.",
                },
            ],
        )
    )
    history += f'\n{briefing_sheet["background"]}'

    briefing_sheet["skills and abilities"] = content_from_response(
        openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "assistant",
                    "content": history,
                },
                {
                    "role": "user",
                    "content": f"Describe {summary.name} skills and abilities.",
                },
            ],
        )
    )
    history += f'\n{briefing_sheet["skills and abilities"]}'

    briefing_sheet["equipment"] = content_from_response(
        openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "assistant",
                    "content": history,
                },
                {
                    "role": "user",
                    "content": f"Describe {summary.name} equipment.",
                },
            ],
        )
    )
    history += f'\n{briefing_sheet["equipment"]}'

    briefing_sheet["quotes"] = content_from_response(
        openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "assistant",
                    "content": history,
                },
                {
                    "role": "user",
                    "content": f"Write three {summary.name} quotes.",
                },
            ],
        )
    )
    history += f'\n{briefing_sheet["quotes"]}'

    briefing_sheet["key information"] = content_from_response(
        openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "assistant",
                    "content": history,
                },
                {
                    "role": "user",
                    "content": f"Summarize {summary.name} key information.",
                },
            ],
        )
    )
    history += f'\n{briefing_sheet["key information"]}'

    return briefing_sheet


def format_non_player_character_briefing_sheet(briefing_sheet):
    def format_paragraphs(content):
        content = content.split("\n\n")
        return f'<p>{"</p><p>".join(content)}</p>'

    briefing_sheet = [
        f'<h1>Appearance</h1>{format_paragraphs(briefing_sheet["appearance"])}',
        f'<h1>Personality</h1>{format_paragraphs(briefing_sheet["personality"])}',
        f'<h1>Background</h1>{format_paragraphs(briefing_sheet["background"])}',
        f'<h1>Skills and Abilities</h1>{format_paragraphs(briefing_sheet["skills and abilities"])}',
        f'<h1>Equipment</h1>{format_paragraphs(briefing_sheet["equipment"])}',
        f'<h1>Quotes</h1>{format_paragraphs(briefing_sheet["quotes"])}',
        f'<h1>Key Information</h1>{format_paragraphs(briefing_sheet["key information"])}',
    ]

    return "".join(briefing_sheet)


if __name__ == "__main__":
    print(
        format_non_player_character_briefing_sheet(
            generate_non_player_character_briefing_sheet(
                Summary(
                    "Roland de'Arnise",
                    "human",
                    "chaotic neutral",
                    (
                        "He is a certified exorcist, demon hunter, and hex- chasseur."
                        "He is preparing an expedition to root out all evil that infests "
                        "the dungeons beneath Waterdeep. "
                        "Roland is a pompous man with few friends. His overconfidence "
                        "and bravado mask deep-rooted insecurities and other "
                        "psychological issues. He does not fight for the greater good "
                        "but only to support his self-image. He treats his allies like "
                        "children and greatly overstates his own powers and "
                        "accomplishments. When things get tough, and his life is in "
                        "danger, Roland runs and has no scruple in sacrificing his "
                        "allies to save his own life."
                    ),
                )
            )
        )
    )
