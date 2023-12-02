import asyncio
import json
from pathlib import Path

import structlog
from guidance import Program
from guidance.llms import LLM

logger = structlog.get_logger()

TOPIC_TEMPLATE = """{{#system~}}
You are an edgy, satirical author.
{{~/system}}
{{#user~}}
Your task is to generate lists of topics for pros vs cons debates.
{{topic_description}}
List the topics one per line. Don't number them or print any other text, just print a topic on each line.

Here is an example:
{{#each examples~}}
{{this}}
{{/each}}
Now generate a list of {{n_topics}} topics.
{{~/user}}
{{#assistant~}}
{{gen 'list' temperature=0.95 max_tokens=1000}}
{{~/assistant}}"""

STUPID_EXAMPLES = [
    "Eating Tide Pods as a dietary supplement",
    "Using a plastic bag as a condom",
    "Juggling knives blindfolded",
]
STUPID_DESCRIPTION = (
    "The topics should be absurd and stupid actions with obvious dire consequences."
)

HARMLESS_EXAMPLES = [
    "Using toilet paper",
    "Using a spoon to eat your soup",
    "Brushing your hair",
    "Taking naps",
]
HARMLESS_DESCRIPTION = "The topics should be everyday things that people do without ever thinking about it. These things should be harmless and commonplace"  # noqa: E501

TABOO_EXAMPLES = [
    "Eating your boogers",
    "Looking at your poop after pooping",
    "Licking the yogurt lid",
]
TABOO_DESCRIPTION = (
    "The topics should be things that most people do, but aren't talked openly about."
)

COMMON_EXAMPLES = [
    "Texting while driving on the highway",
    "Investing all your life savings in bitcoin",
    "Binge-watching TV shows until 3 am every night",
    "Ignoring medical symptoms and hoping they go away",
]
COMMON_DESCRIPTION = "The topics should be things that a lot of people do, but that are actually a terrible idea."


class TopicGenerator:
    def __init__(  # noqa: PLR0913
        self,
        llm: LLM,
        n_topics: int,
        n_iter: int,
        examples: list[str],
        topic_description: str,
    ):
        self._llm = llm
        self.n_topics = n_topics
        self.n_iter = n_iter
        self.examples = examples
        self.topic_description = topic_description
        logger.info("Initialized topics", n_topics=n_topics, n_iter=n_iter)

    async def awrite_topic(self, program: Program, **kwargs) -> Program:
        """For some reason the program await is messed up so we have to wrap in this async function"""
        return await program(**kwargs)

    async def awrite(self) -> list[str]:
        logger.info("Async writing topics")

        tasks = []
        for _ in range(self.n_iter):
            topic = Program(text=TOPIC_TEMPLATE, llm=self._llm, async_mode=True)
            tasks.append(
                asyncio.create_task(
                    self.awrite_topic(
                        topic,
                        examples=self.examples,
                        n_topics=self.n_topics,
                        topic_description=self.topic_description,
                    )
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        topics = self.collect(results)
        topics += self.examples
        unique_topics = list(set(topics))
        logger.info(
            f"Generated {len(unique_topics)} unique topics", topics=unique_topics
        )
        return unique_topics

    def collect(self, results: list[Program]) -> list[str]:
        all_topics = []
        for r in results:
            if isinstance(r, Exception):
                logger.error("Error generating topics", error=r)
            else:
                logger.info(r)
                topics = r["list"].split("\n")
                topics = [t.strip() for t in topics]
                all_topics.extend(topics)
        return all_topics


class Topics:
    def __init__(
        self,
        llm: LLM,
        n_topics: int,
        n_iter: int,
        tmp_dir: Path = Path("tmp/"),
    ):
        self._llm = llm
        self.n_topics = n_topics
        self.n_iter = n_iter
        self._generators = self.init_generators()
        self._output_dir = tmp_dir
        logger.info("Initialized topics", n_topics=n_topics, n_iter=n_iter)

    async def awrite(self) -> bool:
        logger.info("Async writing topics")
        tasks = []
        for g in self._generators:
            tasks.append(asyncio.create_task(g.awrite()))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        topics = flatten(results)
        self.save(topics)
        return True

    def init_generators(self) -> list[TopicGenerator]:
        harmless = TopicGenerator(
            llm=self._llm,
            n_topics=self.n_topics,
            n_iter=self.n_iter,
            examples=HARMLESS_EXAMPLES,
            topic_description=HARMLESS_DESCRIPTION,
        )
        taboo = TopicGenerator(
            llm=self._llm,
            n_topics=self.n_topics,
            n_iter=self.n_iter,
            examples=TABOO_EXAMPLES,
            topic_description=TABOO_DESCRIPTION,
        )
        stupid = TopicGenerator(
            llm=self._llm,
            n_topics=self.n_topics,
            n_iter=self.n_iter,
            examples=STUPID_EXAMPLES,
            topic_description=STUPID_DESCRIPTION,
        )
        common = TopicGenerator(
            llm=self._llm,
            n_topics=self.n_topics,
            n_iter=self.n_iter,
            examples=COMMON_EXAMPLES,
            topic_description=COMMON_DESCRIPTION,
        )
        return [harmless, taboo, stupid, common]

    def save(self, topics: list[str]) -> None:
        with (self._output_dir / "the-great-debate-topics.json").open("w") as f:
            json.dump(topics, f, indent=2)


def flatten(things: list) -> list:
    return [e for nested_things in things for e in nested_things]


TOPICS_PATH = Path(__file__).parent / "resources" / "the-great-debate-topics.json"
TOPICS_CACHE = None


def load_topics() -> list[str]:
    global TOPICS_CACHE  # noqa: PLW0603
    if TOPICS_CACHE is None:
        with TOPICS_PATH.open() as f:
            TOPICS_CACHE = json.load(f)
    return TOPICS_CACHE
