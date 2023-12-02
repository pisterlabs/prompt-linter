import math
from itertools import islice
from typing import AsyncIterable, Optional

import networkx as nx
import nltk
import numpy as np
import openai
import pandas as pd
import tqdm
import typer
from asyncstdlib import map as amap
from asyncstdlib.functools import reduce as areduce
from graphviz import Digraph
from instructor.patch import wrap_chatcompletion
from pydantic import BaseModel, Field
from tenacity import retry, wait_random_exponential

from prompting_techniques import AsyncTyper, async_disk_cache, format_prompt

np.random.seed(1)

nltk.download("punkt")

client = openai.AsyncOpenAI()
func = wrap_chatcompletion(client.chat.completions.create)
app = AsyncTyper()
sent_detector = nltk.data.load("tokenizers/punkt/english.pickle")

ARTICLE = """
Little Red Riding Hood
Story by Leanne Guenther

Once upon a time, there was a little girl who lived in a village near the forest.  Whenever she went out, the little girl wore a red riding cloak, so everyone in the village called her Little Red Riding Hood.

One morning, Little Red Riding Hood asked her mother if she could go to visit her grandmother as it had been awhile since they'd seen each other.

"That's a good idea," her mother said.  So they packed a nice basket for Little Red Riding Hood to take to her grandmother.

When the basket was ready, the little girl put on her red cloak and kissed her mother goodbye.

"Remember, go straight to Grandma's house," her mother cautioned.  "Don't dawdle along the way and please don't talk to strangers!  The woods are dangerous."

"Don't worry, mommy," said Little Red Riding Hood, "I'll be careful."

But when Little Red Riding Hood noticed some lovely flowers in the woods, she forgot her promise to her mother.  She picked a few, watched the butterflies flit about for awhile, listened to the frogs croaking and then picked a few more. 

Little Red Riding Hood was enjoying the warm summer day so much, that she didn't notice a dark shadow approaching out of the forest behind her...

Suddenly, the wolf appeared beside her.

"What are you doing out here, little girl?" the wolf asked in a voice as friendly as he could muster.

"I'm on my way to see my Grandma who lives through the forest, near the brook,"  Little Red Riding Hood replied.

Then she realized how late she was and quickly excused herself, rushing down the path to her Grandma's house. 

The wolf, in the meantime, took a shortcut...

The wolf, a little out of breath from running, arrived at Grandma's and knocked lightly at the door.

"Oh thank goodness dear!  Come in, come in!  I was worried sick that something had happened to you in the forest," said Grandma thinking that the knock was her granddaughter.

The wolf let himself in.  Poor Granny did not have time to say another word, before the wolf gobbled her up!

The wolf let out a satisfied burp, and then poked through Granny's wardrobe to find a nightgown that he liked.  He added a frilly sleeping cap, and for good measure, dabbed some of Granny's perfume behind his pointy ears.

A few minutes later, Red Riding Hood knocked on the door.  The wolf jumped into bed and pulled the covers over his nose.  "Who is it?" he called in a cackly voice.

"It's me, Little Red Riding Hood."

"Oh how lovely!  Do come in, my dear," croaked the wolf.

When Little Red Riding Hood entered the little cottage, she could scarcely recognize her Grandmother.

"Grandmother!  Your voice sounds so odd.  Is something the matter?" she asked.

"Oh, I just have touch of a cold," squeaked the wolf adding a cough at the end to prove the point.

"But Grandmother!  What big ears you have," said Little Red Riding Hood as she edged closer to the bed.

"The better to hear you with, my dear," replied the wolf.

"But Grandmother!  What big eyes you have," said Little Red Riding Hood.

"The better to see you with, my dear," replied the wolf.

"But Grandmother!  What big teeth you have," said Little Red Riding Hood her voice quivering slightly.

"The better to eat you with, my dear," roared the wolf and he leapt out of the bed and began to chase the little girl.

Almost too late, Little Red Riding Hood realized that the person in the bed was not her Grandmother, but a hungry wolf.

She ran across the room and through the door, shouting, "Help!  Wolf!" as loudly as she could.

A woodsman who was chopping logs nearby heard her cry and ran towards the cottage as fast as he could.

He grabbed the wolf and made him spit out the poor Grandmother who was a bit frazzled by the whole experience, but still in one piece."Oh Grandma, I was so scared!"  sobbed Little Red Riding Hood, "I'll never speak to strangers or dawdle in the forest again."

"There, there, child.  You've learned an important lesson.  Thank goodness you shouted loud enough for this kind woodsman to hear you!"

The woodsman knocked out the wolf and carried him deep into the forest where he wouldn't bother people any longer.

Little Red Riding Hood and her Grandmother had a nice lunch and a long chat.
"""


class Node(BaseModel):
    id: int
    label: str
    color: str

    def __hash__(self) -> int:
        return hash((id, self.label))


class Edge(BaseModel):
    source: int
    target: int
    label: str
    color: str = "black"

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.label))


class KnowledgeGraph(BaseModel):
    nodes: Optional[list[Node]] = Field(..., default_factory=list)
    edges: Optional[list[Edge]] = Field(..., default_factory=list)

    def update(self, other: "KnowledgeGraph") -> "KnowledgeGraph":
        """Updates the current graph with the other graph, deduplicating nodes and edges."""
        nodes = self.nodes if self.nodes is not None else []
        edges = self.edges if self.edges is not None else []
        other_nodes = other.nodes if other.nodes is not None else []
        other_edges = other.edges if other.edges is not None else []
        return KnowledgeGraph(
            nodes=list(set(nodes + other_nodes)),
            edges=list(set(edges + other_edges)),
        )

    def draw(self, prefix: Optional[str] = None):
        dot = Digraph(comment="Knowledge Graph")
        nodes = self.nodes if self.nodes is not None else []
        edges = self.edges if self.edges is not None else []

        # Add nodes
        for node in nodes:
            dot.node(str(node.id), node.label, color=node.color)

        # Add edges
        for i, edge in enumerate(edges):
            dot.edge(str(edge.source), str(edge.target), label=f"{i} {edge.label}", color=edge.color)
        dot.render(filename=f"./data/{prefix}", format="png", view=False)


@async_disk_cache(filename="./data/cache.db")
@retry(wait=wait_random_exponential(multiplier=1, max=3))
async def update_kb_graph(graph: KnowledgeGraph, text: str) -> KnowledgeGraph:
    result: KnowledgeGraph = await func(
        messages=[
            {
                "role": "system",
                "content": format_prompt(
                    """
                You are an iterative knowledge graph builder.
                You are given the current state of the graph, and you must append the nodes and edges 
                to it Do not procide any duplcates and try to reuse nodes as much as possible. 
                
                - Ensure that the only nodes are characters of the story
                - Ensure that edges are significant interactions between characters
                - Do not repeat nodes or edges of characters or interactions that have already been added
                - Ignore everything else
                """
                ),
            },
            {
                "role": "user",
                "content": f"""Extract any new nodes and edges from the following:

                {text}""",
            },
            {
                "role": "user",
                "content": f"""Here is the current state of the graph:
                {graph.model_dump_json(indent=2)}""",
            },
        ],
        model="gpt-4",
        response_model=KnowledgeGraph,
        temperature=0,
        seed=256,
    )
    kb = KnowledgeGraph.model_validate(graph.update(result))
    kb.draw("knowledge_graph")
    return kb


async def sliding_window(iterable: list[str], window_size: int, stride: int) -> AsyncIterable[list[str]]:
    """Generate a sliding window of specified size over the iterable."""
    total_iterations = math.ceil((len(iterable) - window_size) / stride) + 1
    with tqdm.tqdm(desc="Sliding Window", total=total_iterations) as progress:    
        for i in range(0, len(iterable)-window_size+1, stride):
            yield iterable[i:i+window_size]
            progress.update(1)

@app.command()
async def kb_generation():
    typer.echo("Running reduce based summary on news article.")
    typer.echo("\n")

    window_size = 5
    stride = 4
    sentences: list[str] = sent_detector.tokenize(ARTICLE)  # type: ignore
    async def join_sentences(sentences: list[str]) -> str: return " ".join(sentences)
    kb = await areduce(update_kb_graph, amap(join_sentences, sliding_window(sentences, window_size, stride)), initial=KnowledgeGraph()) # type: ignore
    kb.draw("knowledge_graph") # type: ignore

if __name__ == "__main__":
    app()
