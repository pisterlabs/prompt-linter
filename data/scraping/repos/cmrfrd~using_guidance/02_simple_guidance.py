import random
from itertools import accumulate
from operator import itemgetter
from typing import Generator, Iterable

from guidance import Program, llms


def empty_get(d: dict, key: str) -> str:
    """
    Returns the value of a key in a dictionary, or an empty string if the key is not present.

    Args:
        d: The dictionary to get the value from.
        key: The key to get the value for.

    Returns:
        The value of the key in the dictionary, or an empty string if the key is not present.
    """
    return d.get(key, "")


def generate_story(name: str) -> Generator[str, None, None]:
    """
    Generates a story for a given name.

    Args:
        name: The name of the person in the story.

    Yields:
        A stream of story content from the llm.
    """

    sentiments: list[str] = ["happy", "sad", "excited", "angry", "gratitude"]
    base_template = """
    Write a 3 sentence story about a person and use {{input_name}} whenever you need to write their name.
    
    Sentiment: {{sentiment}}
    
    Story:
    {{gen max_tokens=64 name="story"}}
    """

    # define a guidance program that adapts a proverb
    iprogram: Generator[Program, None, None] = Program(
        base_template,
        llm=llms.OpenAI("text-davinci-003"),
        caching=False,
        stream=True,
        silent=True,
    )(
        input_name=name,
        sentiment=random.choice(sentiments),
    )

    # Only retrieve the "story" field from the program's output
    story_iter: Iterable[str] = map(
        lambda p: empty_get(p, "story"), iprogram  # pylint: disable=protected-access
    )

    # Guidance only outputs the full story at every iteration
    # only yield the new content of the story
    yield from map(
        itemgetter(1),
        accumulate(story_iter, lambda x, y: (len(y), y[x[0] :]), initial=(0, "")),
    )


for chunk in generate_story("Alex"):
    print(chunk, end="")
