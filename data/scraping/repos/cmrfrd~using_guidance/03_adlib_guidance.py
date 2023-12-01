import re
from enum import Enum
from functools import partial, reduce
from itertools import accumulate
from operator import itemgetter
from typing import Generator, Iterable, Optional

from guidance import Program, llms
from pydantic import BaseModel


def clean_string(string: str) -> str:
    """Remove whitespace and newlines."""
    return string.strip().replace("\n", "")


def fill_madlib(mood: str) -> str:
    """ """

    base_template = """
{{#block hidden=True}}
# Setup

Here is an adlib story:

It was ___(FOOD 1)___ day at school, and ___(NAME 1)___ was super ___(ADJECTIVE 1)___ for lunch. 
But when she went outside to eat, a ___(NOUN 1)___ stole her ___(FOOD 1)___! ___(NAME 1)___ chased 
the ___(NOUN 1)___ all over school. She ___(VERB 1)___, ___(VERB 2)___, and ___(VERB 3)___ through 
the playground. Then she tripped on her ___(NOUN 2)___ and the ___(NOUN 1)___ escaped! Luckily, ___(NAME 1)___’s 
friends were willing to share their ___(FOOD 2)___ with her.

# Goal

You are an AI mad lib completor. Please output a single word after each listed part of speech. Try to choose
the words in the mood of {{mood}}.

- Food 1: {{gen 'food_1' max_tokens=1}}
- Person Name 1: {{gen 'name_1' max_tokens=1}}
- Adjective 1: {{gen 'adjective_1' max_tokens=2}}
- Noun 1: {{gen 'noun_1' max_tokens=2}}
- Past Verb 1: {{gen 'verb_1' max_tokens=2}}
- Past Verb 2: {{gen 'verb_2' max_tokens=2}}
- Past Verb 3: {{gen 'verb_3' max_tokens=2}}
- Noun 2: {{gen 'noun_2' max_tokens=2}}
{{/block}}

{{#block name="story"}}
It was {{clean_string food_1}} day at school, and {{clean_string name_1}} was super {{clean_string adjective_1}} for lunch. 
But when she went outside to eat, a {{clean_string noun_1}} stole her {{clean_string food_1}}! {{clean_string name_1}} chased 
the {{clean_string noun_1}} all over school. She {{clean_string verb_1}}, {{clean_string verb_2}}, and {{clean_string verb_3}} through 
the playground. Then she tripped on her {{clean_string noun_2}} and the {{clean_string noun_1}} escaped! Luckily, {{clean_string name_1}}’s 
friends were willing to share their {{clean_string food_1}} with her.
{{/block}}
"""

    program: Program = Program(base_template, llm=llms.OpenAI("text-davinci-003"))(
        caching=False, silent=True, mood=mood, clean_string=clean_string
    )()
    return str(program.text)


print(fill_madlib("emotional"))
