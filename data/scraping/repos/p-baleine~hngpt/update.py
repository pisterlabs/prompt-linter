import click
import datetime
import functools
import logging
import orgparse
from langchain.chains.base import Chain
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from tqdm.auto import tqdm
from typing import Optional

from hngpt.chain import Review, ReviewerChain
from hngpt.hnclient import HackerNewsStory, get_hn_topstories


logger = logging.getLogger(__name__)


def find_org_entry(
        node: orgparse.OrgNode,
        hacker_news_id: str,
        property_key: str = "Hacker_News_ID",
) -> Optional[orgparse.OrgNode]:
    return [
        n for n in node.children
        if int(n.get_property(property_key)) == hacker_news_id
    ]


def create_new_entry_output(
        summarize_chain: Chain,
        story: HackerNewsStory,
        review: Review,
        level: int = 1,
        review_threshold_for_summarization: int = 1
) -> str:
    def format_org_timestamp(datetime: datetime.datetime):
        return datetime.strftime("<%Y-%m-%d %H:%M>")

    now = datetime.datetime.now()
    indent = "  " * level
    result = f"""{'*' * level} [[{story.url}][{story.title}]]
{indent}:PROPERTIES:
{indent}:Title: {story.title}
{indent}:Hacker_News_ID: {story.id}
{indent}:Posted_at: {format_org_timestamp(story.posted_at)}
{indent}:Review_score: {review.score}
{indent}:Reviewed_at: {format_org_timestamp(now)}
{indent}:Review_comment: {review.reason}
{indent}:END:
"""

    result += f"""
{'*' * (level + 1)} Review
- Review score :: {review.score}
- Reviewed at :: {format_org_timestamp(now)}
- Review comment :: {review.reason}
"""

    if story.documents:
        summary = summarize_chain.run([story.documents[0]])
        result += f"""{'*' * (level + 1)} Summary
{summary}
"""

    return result


@click.command()
@click.option("--org-path", type=click.Path(exists=True))
@click.option("-n", type=click.INT, default=20)
@click.option("--verbose", type=click.BOOL, default=False)
def main(org_path, n, verbose):
    root = orgparse.load(org_path)
    llm = ChatOpenAI(temperature=0)
    reviewer_chain = ReviewerChain(llm=llm, verbose=verbose)
    summarize_chain = load_summarize_chain(llm, chain_type="stuff", verbose=verbose)
    _create_new_entry_output = functools.partial(
        create_new_entry_output,
        summarize_chain=summarize_chain,
    )
    new_entries = []

    logger.info("Fetching HN top stories.")

    with get_openai_callback() as cb:
        for story in tqdm(get_hn_topstories(n=n)):
            if find_org_entry(root, story.id):
                continue

            logger.info(f"Review `{story.title}` by GPT.")

            review = reviewer_chain.run(story=story)
            new_entries.append(_create_new_entry_output(
                story=story,
                review=review
            ))

    with open(org_path, "a") as f:
        for entry in new_entries:
            f.write("\n")
            f.write(entry)

    logger.info(f"Token usage: {cb.total_tokens} tokens.")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.getLogger("hngpt").setLevel(level=logging.DEBUG)
    main()
