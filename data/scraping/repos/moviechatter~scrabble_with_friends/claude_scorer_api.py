from utils import fetch_and_clean_html

import re
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chat_models import ChatAnthropic
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

ANTHROPIC_API_KEY = os.environ['ANTHROPIC_API_KEY']

app = FastAPI()

class Url(BaseModel):
    url: str

class ScorePayload(BaseModel):
    context: str
    word: str

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/summarize_webpage")
async def webpage_summarize(link: Url):
    url = link.url

    cleaned_html = fetch_and_clean_html(url)

    chat = ChatAnthropic(
        model="claude-2.0",
        temperature=0.0,
        top_p=0.95,
        anthropic_api_key=ANTHROPIC_API_KEY
    )

    chain = ConversationChain(
        llm=chat,
        memory=ConversationBufferMemory()
    )

    summarizer = f"""Here is a document, in <document></document> XML tags. This document is cleaned html body text extracted from {url}: <document>\n${cleaned_html}</document>\n\Summarize the key details such as the main topic, products/services described, and company name in an executive summary written in professional tone. If it is a products listing, also list the most relevant products in this webpage."""

    summary = chain.run(summarizer)
    print(summary)

    ideas_themes = chain.run("List 'Main Ideas', 'Potential SEO Keywords', and 'Relevant Topics' of the document content. 'Relevant Topics' should be a compilation of topics that appropriately categorizes this document content. Even with limited context, extrapolate to the best of your ability.")

    export_context = f"""{summary}\n{ideas_themes}"""

    with open("context.txt", "w") as file:
        file.write(export_context)

    return f"""{summary}\n{ideas_themes}"""


@app.post("/get_score")
def relevance_score(score_json: ScorePayload):
    chat = ChatAnthropic(
        model="claude-2.0",
        temperature=0.0,
        top_p=0.95,
        anthropic_api_key=ANTHROPIC_API_KEY
    )
    should_score_prompt = f"""You are a casual Scrabble player who wants to win. You are given the following information:
Executive Summary, Main Ideas, Themes:
<context>
{score_json.context}
</context>

Word: {score_json.word}

Answer "True" or "False" to the following statement: For this word, I can award bonus points for a version of Scrabble that awards points for terms that are even semi relevant to the context.
"""
    print(should_score_prompt)

    criteria_prompt = f"""You are a casual Scrabble player who wants to win. You follow the following criteria when scoring the relevancy of the word above:
1. Keyword Matching (30%) - Exact match = 6 points, Partial match = 3 points, No match = 0 points
2. Semantic Relevance (20%) - Highly related = 4 points, Moderately related = 2 points, Not related = 0 points
3. Word Relationships (15%) - Direct strong relationship = 3 points, Moderate relationship = 2 points, No relationship = 0 points
4. Creativity/Novelty/Cleverness (20%) - Highly creative = 4 points, Moderately creative = 2 points, Not creative = 0 points
5. Intent Analysis (15%) - Strongly matches = 3 points, Moderately matches = 2 points, Weakly matches = 1 point, Doesn't match = 0 points
Award points liberally, but make it difficult to get full marks unless the word is highly relevant. Only output score in format "Total Points: [insert score here]/20"""
    # and a very concise one-sentence reasoning per criteria followed by total point value

    chain = ConversationChain(
        llm=chat,
        memory=ConversationBufferMemory()
    )

    should_score = chain.run(should_score_prompt)
    print(should_score)

    if "True" in should_score:
        scoring = chain.run(criteria_prompt)
        print(scoring)
        points_regex = r"Points: ([\d]+)\/20"
        points_matches = re.search(points_regex, scoring)
        score = int(points_matches.group(1))
    else:
        score = int(0)
    return score
