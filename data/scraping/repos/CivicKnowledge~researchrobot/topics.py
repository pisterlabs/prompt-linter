import json
import logging
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import List, Union

import pandas as pd

from researchrobot.openai.completions import openai_one_completion

logger = logging.getLogger(__name__)


@dataclass
class Topic:
    path: List[str] = field(default_factory=list)
    description: str = ""
    children: List[Union["Topic", str]] = field(default_factory=list)

    @property
    def parent_path(self):
        if not self.path:
            return None
        return self.path[:-1]

    @property
    def parent_path_str(self):
        if not self.path:
            return ""
        return "/".join(self.path[:-1])

    @property
    def path_str(self):
        return "/".join(self.path)


def _load_file():
    """Read the categories yaml file from the support module"""
    import yaml

    import researchrobot.datadecomp.support as rs

    rsp = Path(rs.__file__).parent
    cat_p = rsp / "categories.yaml"

    with cat_p.open() as f:
        cats = yaml.safe_load(f)

    return cats


def _walk_data(key, value, path, f, memo):
    """Recursively walk the data structure, calling f on each node"""

    key = key.lower() if key else None

    if isinstance(value, dict):

        # This case is for the description of a leaf. Non-leaves have a '_description' entry,
        # but for leaves, there is just a string value for a single entry dict.
        first_k, first_v = list(value.items())[0]
        if isinstance(first_v, str) and not first_k.startswith("_"):
            value = {first_k: {"_description": first_v}}

        for k, v in value.items():
            if f(memo, key, value, path + [k]):
                _walk_data(k, v, path + [k], f, memo)
    elif isinstance(value, list):
        for v in value:
            _walk_data(key, v, path, f, memo)
    elif isinstance(value, str):
        f(memo, key, None, path + [value])

    return memo


def _compile_topics(memo, key, value, path):
    path = tuple(path)

    if path[-1].startswith("_"):
        if path[-1] == "_see_also":
            return False
        elif path[-1] == "_description":
            path_str = "/".join(path[:-1])
            memo[path_str].description = value["_description"]

        # print('Desc', path[:-1],  value['_description'])

        return False
    else:

        if path in memo:
            print("ERR duplicate", path)

        topic = Topic(path)

        # print('  '*len(path),topic)

        memo[topic.path_str] = topic
        if topic.parent_path_str:
            memo[topic.parent_path_str].children.append(topic)
            # print(('  '*len(path)),'- add_child to: ', topic.parent_path)

    return True


def load_categories():
    """Load the categories from the module, then convert to a hierarchy of Topic objects"""

    cats = _load_file()

    # Walk the data structure, creating a Topic object for each path
    topics = _walk_data(None, cats, [], _compile_topics, {})

    return topics


def add_other(data):
    if isinstance(data, dict):
        for value in data.values():
            add_other(value)
    elif isinstance(data, list):
        if "other" not in data:
            data.append("other")
        for item in data:
            add_other(item)


def compute_paths(level, acc=[], base=""):
    """Given a nested representation, return the paths."""

    if isinstance(level, dict):
        for k, v in level.items():
            compute_paths(v, acc, base + "/" + k)
    elif isinstance(level, (list, tuple)):
        for v in level:
            compute_paths(v, acc, base)
    elif isinstance(level, str):
        acc.append(base + "/" + level)
    else:
        assert False, f"unknown type {level}"

    return acc


refine_prompt_templ = """
For the provided question, return the numbers of the topics which best describe the {object_type}.
A {object_type} may have multiple topics, each for a different aspect of the {object_type}. For instance,
a {object_type} about "35 year old white married women " could be classified with the topics of
'people/age', 'people/race', 'people/sex', and 'people/family/marriage'.

You will be provided with a subset of topics generated from a semantic search. Apply only the
topics in this list. Each line in the list will have: the topic index number, the topic path,
a ":", then and optional description. If the description is not given, infer the description
from the topic path.

Return your response as a JSON list of index numbers. For most questions, more than one topic
will apply. If you return more than one topic, the topics should be in order of importance.

# Topics

{topics}

# {object_type}

{question}

# JSON List Response

"""


def results_str(df):
    """Format a topics search results string for a prompt"""

    out = ""

    for idx, r in df.iterrows():
        out += f"   {idx} {r.path}: {r.description}\n"

    return out


class TopicCategorizer:
    def __init__(self):
        from researchrobot.embeddings import EmbedDb

        self.topics = load_categories()

        self.db = EmbedDb("topics")

    def load_database(self):
        """Load the topics into the Vector database"""

        self.db.drop_collection()
        self.db.load_collection(self.topics_df)

    @cached_property
    def topics_df(self):

        rows = []

        for k, v in self.topics.items():
            rows.append(
                {
                    "text": f"{v.path_str}: {v.description}"
                    if v.description
                    else v.path_str,
                    "path": v.path_str,
                    "description": v.description,
                }
            )

        return pd.DataFrame(rows)

    def search(self, q):
        e = self.db.vector_query([q], limit=20)
        e = e.sort_values(["score"], ascending=False)

        st = self.topics_df[self.topics_df.path.str.startswith("survey_types/")]

        # Rebuild candidate topics including the survey types
        e = pd.concat(
            [st, self.topics_df[self.topics_df.path.isin(e.path)]]
        ).drop_duplicates()

        return e

    def refine(self, q, results, return_response=False):

        prompt = refine_prompt_templ.format(
            object_type="question", topics=results_str(results), question=q
        )

        r = openai_one_completion(prompt)

        try:
            idx = json.loads(r)
        except Exception as e:
            logger.error(f"Error parsing response: {r} Exception='{str(e)}'")
            raise

        df = (self.topics_df.loc[idx]).rename(columns={"path": "topic"})

        if return_response:
            return df, r
        else:
            return df

    def search_refine(self, q, return_response=False):
        e = self.search(q)
        return self.refine(q, e, return_response)
