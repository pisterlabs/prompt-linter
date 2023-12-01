import asyncio
from itertools import islice
from typing import AsyncIterable, Generator, Iterable

import nltk.data
import numpy as np
import openai
import pandas as pd
import tqdm
import typer
from asyncstdlib import map as amap
from asyncstdlib.functools import reduce as areduce
from tenacity import retry, wait_random_exponential

from prompting_techniques import AsyncTyper, async_disk_cache, execute, format_prompt

np.random.seed(1)
nltk.download('punkt')

client = openai.AsyncOpenAI()
app = AsyncTyper()
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

ARTICLE = """
OpenAI's board of directors' abruptly firing CEO Sam Altman then bringing him back days later did not come out of nowhere.

In fact, the boardroom drama represented the boiling over of tensions that have long simmered under the surface of the company.

Following days of upheaval, Altman is again leading the company and a newly-formed board of directors is charting the path ahead, but the chaos at OpenAI can be traced back to the unusual way the company was structured.

OpenAI was founded in 2015 by Altman, Elon Musk and others as a non-profit research lab. It was almost like an anti-Big Tech company; it would prioritize principles over profit. It wanted to, as OpenAI put it back then, develop AI tools that would "benefit humanity as a whole, unconstrained by a need to generate financial return."

With his sudden return to OpenAI, Sam Altman becomes the latest 'boomerang CEO'
BUSINESS
With his sudden return to OpenAI, Sam Altman becomes the latest 'boomerang CEO'
But in 2018, two things happened: First, Musk quit the board of OpenAI after he said he invested $50 million, cutting the then-unknown company off from more of the entrepreneur's crucial financial backing.

And secondly, OpenAI's leaders grew increasingly aware that developing and maintaining advanced artificial intelligence models required an immense amount of computing power, which was incredibly expensive.

Balancing ideals with the need for funding
A year after Musk left, OpenAI created a for-profit arm. Technically, it is what's known as a "capped profit" entity, which means investors' possible profits are capped at a certain amount. Any remaining money is re-invested in the company.

Yet the nonprofit's board and mission still governed the company, creating two competing tribes within OpenAI: adherents to the serve-humanity-and-not-shareholders credo and those who subscribed to the more traditional Silicon Valley modus operandi of using investor money to release consumer products into the world as rapidly as possible in hopes of cornering a market and becoming an industry pacesetter.

Altman, a 38-year-old techno-optimist who previously led the prestigious startup accelerator Y Combinator, tried to thread the needle between the two approaches. He struck something of a middle ground by unveiling new OpenAI tools gradually, first to smaller groups, then larger ones, to fine-tune and refine the tools before making them public.

ChatGPT's success attracts Big Tech money
When OpenAI kicked off a seismic shift in the tech industry with its launch of ChatGPT last year, the company's most prominent investor, Microsoft, greatly increased its financial stake. It upped its commitment to OpenAI to the tune of $13 billion.

Microsoft became the financial engine that powered OpenAI, but the nonprofit's board of directors still called all the shots. Despite Microsoft's sizable investment, it did not have a seat on OpenAI's board.

All of this set the stage for Altman's sudden ouster from the company earlier this month.

The board itself has still not explained why it fired Altman — beyond saying, in vague terms, that it believed Altman had not been "consistently candid in his communications with the board." And the company's structure gives the board that right: it has complete, unchecked power to remove the CEO whenever it sees fit.

Sources close to the discussions say before Altman's termination, he had been at odds with members of the board over the hasty commercialization of OpenAI products. Board members worried whether Altman was considering the risks of AI products seriously enough, or just trying to maintain the company's dominant position in the crowded and competitive world of generative AI development.

The dangers of powerful AI range from supercharging the spread of disinformation, massive job loss and human impersonation exploited by bad actors.

The question was, did Altman abandon OpenAI's founding principles to try to scale up the company and sign up customers as fast as possible? And, if so, did that make him unsuited to helm a nonprofit created to develop AI products "free from financial obligations"?

Whatever its reasoning, there was nothing Microsoft, or any company executive, could do when the board moved to jettison Altman. The dramatic gesture, and then reversal, illustrated the tension at the heart of OpenAI's structure.

An anonymous letter written by former OpenAI employees during the Altman drama called on the board to examine whether Altman was putting commercial products and fundraising goals before the nonprofit's founding mission.

"We implore you, the Board of Directors, to remain steadfast in your commitment to OpenAI's original mission and not succumb to the pressures of profit-driven interests," the letter states. "The future of artificial intelligence and the well-being of humanity depend on your unwavering commitment to ethical leadership and transparency."

An uneasy resolution
OpenAI's board at first refused to entertain the possibility of Altman returning, but then something happened they could not ignore: 702 out of OpenAI's 770 employees committed to leaving the company unless Altman was restored. The employees also asked that a new board be assembled. It was, and Altman was restored as CEO not long after.

Just one former board member sits on the new, temporary board: Adam D'Angelo, the CEO of the question-and-answer site Quora. He had voted for Altman's ouster.

Others, who are familiar to Silicon Valley boards, have taken seats alongside him. They include Bret Taylor, a longtime Silicon Valley executive and former chairman of the board of Twitter, and former Treasury Secretary Larry Summers.

As it stands, OpenAI's charter says it is committed to the development of artificial general intelligence, also known as AGI, or a type of AI superintelligence that can outperform humans, that will not "harm humanity or unduly concentrate power."

But success in Silicon Valley almost always requires massive scale and the concentration of power — something that allowed OpenAI's biggest funder, Microsoft, to become one of the most valuable companies in the world. It is hard to imagine Microsoft would invest $13 billion into a company believing it would not one day have an unmovable foothold in the sector.

Under the board's current mission, developing AI systems should be undertaken with the main goal of benefiting all of humanity, with no regard to ever turning a profit for outside investors.

Yet the for-profit entity of OpenAI will continue to recruit moneyed enthusiasts who want in on the AI goldrush. The two sides are at cross purposes, with no clear way to co-exist.

The new board is expected to grow and include a representative from Microsoft. Among the board's tasks: taking a hard look at OpenAI's structure. Does the hybrid model create too much friction? Or is there a way to forge ahead with a middle-of-the-road approach?
"""

async def sliding_window(iterable: list[str], size: int) -> AsyncIterable[list[str]]:
    """Generate a sliding window of specified size over the iterable."""
    iterators = [list(islice(iterable, i, None)) for i in range(size)]
    
    ## progress
    with tqdm.tqdm(desc="Sliding Window", total=len(iterable)) as progress:
        for item in zip(*iterators):
            yield item
            progress.update(1)


@async_disk_cache(filename="./data/cache.db")
@retry(wait=wait_random_exponential(multiplier=1, max=3))
async def condense(summary: str, new_information: str) -> str:
    """Condense a summary with new information."""
    response = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": format_prompt(
                    f"""
                    You are an AI summarizer. You have one goal: to condense a summary of a news article with new information from the article.

                    Guidelines
                    - Make every word count: Rewrite the previous summary to improve flow and make space for additional entities
                    - Ensure key details are not lost: The new summary should contain all the entities from the previous summary
                    - Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
                    - The new summary should be highly dense and concise yet self-contained, eg., easily understood without the Article.
                    - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses"
                    - Missing entities can appear anywhere in the new summary
                    - Keep the summary at a maximum of 4-5 sentences long

                    An Entity is a real-world object that's assigned a name - for example, a person, country a product or a book title.
                    """
                ),
            },
            {
                "role": "user",
                "content": format_prompt(
                    f"""
                    Here is the previous summary along with the new information is provided below.
                    
                    Here is the summary: {summary}
                    
                    Here is the new information: {new_information}

                    Please output a new condensed summary and nothing else.
                    """
                ),
            }
        ],
        max_tokens=30*6,
        temperature=0.9,
        seed=256,
        model="gpt-3.5-turbo-0613",
    )    
    assert len(response.choices) > 0, "No choices were returned."
    content = response.choices[0].message.content
    assert content is not None, "No text was provided."
    return content

@app.command()
async def reduce_example():
    typer.echo("Running reduce based summary on news article.")
    typer.echo("\n")

    window_size = 4
    sentences: list[str] = sent_detector.tokenize(ARTICLE) # type: ignore
    async def join_sentences(sentences: list[str]) -> str: return " ".join(sentences)
    summary = await areduce(condense, amap(join_sentences, sliding_window(sentences, window_size)), initial="")
    typer.echo(f"Summary: {summary}")
    

if __name__ == "__main__":
    app()
