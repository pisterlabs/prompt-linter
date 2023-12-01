from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from dotenv import load_dotenv
import os

load_dotenv()

anthropic = Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

# The double curly bracket '{{' at the end escapes to be a single JSON '{'
my_prompt = """
{HUMAN_PROMPT} I will provide you with an article about a contentious issue:

<article>{ARTICLE_TEXT}</article>

Please provide up to 10 statements derived from the article, with the following additional rules:
- statements should be the most controversial or divisive from the source material,
- if you can't return 10 statements, you may return fewer. do not invent extra statements not derived from article.
- statements that are complex and contain multiple sentiments or causality, should be returned as multiple statements.
- statements should be short and standalone.
- statements should be as concise as possible.
- each statement should be no longer than 300 characters.
- each statement should try to capture the same tone as the the source sentences.
- each statement should be rephrased into the first person (e.g., "I feel..."), so that a reader can easily determine if it reflects their own opinion.
- the response should be a valid JSON array of objects, with no other text after.
- the response JSON objects should have a "statement" key, and nothing else.

{AI_PROMPT} [{{
"""

# - Favour starting statements with "I feel..." when there's otherwise no personal pronoun.

ARTICLE_TEXT = open("article.txt", "r").read()

completion = anthropic.completions.create(
    model="claude-2",
    max_tokens_to_sample=300,
    prompt=my_prompt.format(ARTICLE_TEXT=ARTICLE_TEXT, HUMAN_PROMPT=HUMAN_PROMPT, AI_PROMPT=AI_PROMPT)
)
print(completion.completion)
