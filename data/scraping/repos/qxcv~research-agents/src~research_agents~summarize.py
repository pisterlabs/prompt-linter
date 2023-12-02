"""Summarize a given paper based on abstract and intro."""
import research_agents.html_to_ref_list as h2rl
from fuzzywuzzy import process
from anthropic import Client
import anthropic
from typing import Optional

TITLE_MAX_LEN = 1000
ABSTRACT_MAX_LEN = 1500
INTRO_MAX_LEN = 2500


def truncate(str: str, max_len: int) -> str:
    trunc_str = "... (truncated)"
    if len(str) > max_len:
        return str[:max_len - len(trunc_str)] + trunc_str
    return str


SUMMARY_TEMPLATE = """{paper_description}

---

Given the paper description above, fill out the following template, formatted as
Markdown (complete the parts in [square brackets]). Respond with _only_ the
formatted Markdown and no other text.

**Title:** [title of the paper]

**What does:** [concise description of what the authors claim that the paper does; 1-2 sentences]

**Why the authors think it's important:** [what does this paper contribute to our scientific understanding of machine learning? How could it change the way that researchers think? 1-2 sentences]

**Headline results:** [most important experimental or theoretical results claimed by the paper; 1-2 sentences]"""

def summarise_paper_md(client: Client, html_str: str) -> str:
    """Given a HTML string from ar5iv.org, return the introduction of the paper,
    formatted as Markdown."""
    title, abstract = h2rl.get_title_and_abstract(html_str)
    paper_description = f"Title: {title}\n\nAbstract: {abstract}"
    sections = h2rl.get_kv_sections(html_str)
    result = process.extractOne('Introduction', sections.keys(), score_cutoff=50)
    if result is not None:
        intro_text = sections[result[0]]
        paper_description += f"\n\nIntroduction: {intro_text}"

    formatted_template = SUMMARY_TEMPLATE.format(paper_description=paper_description)

    resp = client.completion(
        prompt=f"{anthropic.HUMAN_PROMPT} {formatted_template}{anthropic.AI_PROMPT}",
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-1",
        max_tokens_to_sample=500,
        temperature=0
    )

    md_result = resp["completion"]

    return md_result

def format_snippets(snippets):
    """Format snippets in a Markdown-esque way"""
    parts = []
    for idx, snip in enumerate(snippets, start=1):
        parts.append(f"- Snippet {idx}: {snip}")
    return "\n".join(parts)

SNIPPET_SUMMARY_TEMPLATE = """Here is a summary of the original paper:

Original paper title: {original_title}

Abstract: {original_abstract}

-------

Here is a summary from a follow-up paper:

Follow-up title: {follow_up_title}

Follow-up abstracts: {follow_up_abstract}

In this follow-up paper, the original paper may be referred to by name ({original_title}), or by an inline citation style ({follow_up_inline}).
Here are snippets where the follow-up paper mentions the original paper:

{follow_up_snippets}

-----

Can you summarize the new information that the follow-up paper reveals about the original paper? This should be information that is not present in the \
original paper's abstract, but is mentioned in the follow-up paper's abstract or snippets. Provide your answer as dot points that \
quote the follow-up paper. Make a summary, don't use just direct quotes. Don't assume the reader knows about the snippets (VERY IMPORTANT: do not mention the snippets in the dot points). Filter out \
dot points that do not point out differences between the original paper and the follow-up paper."""

def summarize_differences(client, original_paper_html: str, follow_up_html: str) -> Optional[str]:
    title_original, abstract_original = h2rl.get_title_and_abstract(original_paper_html)
    ref_title, ref_abstract = h2rl.get_title_and_abstract(follow_up_html)
    refs = h2rl.get_refs(follow_up_html)
    if len(refs) == 0:
        print(f'No references found')
        return None
    matching_ref = h2rl.find_ref_for(title_original, refs)
    # print(f'Matching ref: {matching_ref}')

    ref_formatted_snippets_about_original = format_snippets(matching_ref["snippets"])
    ref_inline_style = matching_ref["inline"].strip()

    # For each relevant paper, extract a comparison summary (from titles + abstracts + snippets above)
    prompt = SNIPPET_SUMMARY_TEMPLATE.format(
        original_title=title_original,
        original_abstract=abstract_original,
        follow_up_title=ref_title,
        follow_up_abstract=ref_abstract,
        follow_up_inline=ref_inline_style,
        follow_up_snippets=ref_formatted_snippets_about_original,
    )

    resp_ref = client.completion(
        prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-1",
        max_tokens_to_sample=1000,
        temperature=0
    )
    return resp_ref["completion"]


def make_meta_summary(client, responses, title_original) -> str:
    template = """
        The following are comparisons of the original paper {original_title} with follow-up papers. Can you summarize all these dot points into a single paragraph?
        Reply in nicely-formatted Markdown with special formatting for titles and different paragraphs.
        """

    prompt = template.format(original_title=title_original)
    for resp in responses:
        prompt += '\n' + resp

    resp = client.completion(
        prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-1",
        max_tokens_to_sample=500,
        temperature=0
    )

    return resp["completion"]