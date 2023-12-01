import asyncio
import aiohttp
import openai
import os
import spacy
import streamlit as st
import nest_asyncio
import urllib

import css
import prompts

from serpapi import GoogleSearch
from sentence_transformers import CrossEncoder
from claim import Claim
from types import SimpleNamespace


semantic_scholar_api_key = os.environ["semantic_scholar_api_key"]
serpapi_api_key = os.environ["serpapi_api_key"]
openai_api_key = os.environ["openai_api_key"]

openai.api_key = openai_api_key

semantic_scholar_headers = {"x-api-key": semantic_scholar_api_key}
openai_headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai_api_key}",
}


@st.experimental_singleton
def get_msmarco_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512)


@st.experimental_singleton
def get_spacy_nlp():
    try:
        os.system("python -m spacy download en_core_web_sm")
    except Exception as e:
        print(f"Failed to download en_core_web_sm")
        print(e)
    return spacy.load("en_core_web_sm")


msmarco_encoder = get_msmarco_encoder()
spacy_nlp = get_spacy_nlp()


async def openai_instruct_query(session, prompt):
    engine = "davinci-instruct-beta-v2"
    data = {"prompt": prompt, "max_tokens": 600, "temperature": 0}
    response = await session.post(
        f"https://api.openai.com/v1/engines/{engine}/completions",
        json=data,
        headers=openai_headers,
    )
    completion_result = await response.json()
    return completion_result["choices"][0]["text"].strip()


async def answer_to_question(session, claim, question):
    prompt = prompts.answer_to_subquestion.format(answer=claim.text, question=question)
    question = await openai_instruct_query(session, prompt)
    return claim, question


async def abstract_to_claims(session, text):
    prompt = prompts.abstract_to_claims.format(text=text[-2000:])
    result = await openai_instruct_query(session, prompt)
    return [line.strip("- ") for line in result.split("\n")]


@st.cache(persist=True, show_spinner=False, allow_output_mutation=True, max_entries=30)
def score_claims_openai(question, claims):
    documents = [claim.text for claim in claims]
    results = openai.Engine(id="babbage-search-index-v1").search(
        documents=documents, query=question, version="alpha"
    )
    return [(datum["score"], claim) for (datum, claim) in zip(results["data"], claims)]


@st.cache(persist=True, show_spinner=False, allow_output_mutation=True, max_entries=200)
def score_claim_msmarco(question, claim):
    scores = msmarco_encoder.predict([(question, claim.text)])
    return scores[0]


def score_claims(question, claims):
    for claim in claims:
        claim.score = score_claim_msmarco(question, claim)
    sorted_claims = sorted(claims, reverse=True, key=lambda claim: claim.score)
    return sorted_claims


async def scholar_result_to_claims(session, scholar_result):

    cache_key = scholar_result.get("link", scholar_result["title"])
    if cache_key in st.session_state.claims:
        claims = st.session_state.claims[cache_key]
        return claims

    def cache(*values):
        st.session_state.claims[cache_key] = values
        return values

    title = scholar_result.get("title")
    if not title:
        return cache([], "No title found")
    params = {"query": title}
    response = await session.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params=params,
        headers=semantic_scholar_headers,
    )
    response_json = await response.json()
    data = response_json.get("data")
    if not data:
        return cache([], title)
    paper_id = data[0].get("paperId")
    if not paper_id:
        return cache([], title)
    params = {"fields": "title,abstract,venue,authors,citationCount,url,year"}
    paper_detail_response = await session.get(
        f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}",
        params=params,
        headers=semantic_scholar_headers,
    )
    paper_detail = await paper_detail_response.json()
    abstract = paper_detail.get("abstract")
    if not abstract:
        return cache([], title)
    conclusions = await abstract_to_claims(session, text=abstract)
    claims = []
    for conclusion in conclusions:
        claims.append(Claim(text=conclusion, source=paper_detail))
    return cache(claims, title)


async def async_scholar_results_to_claims(
    scholar_results, set_progress, set_claims_view
):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for scholar_result in scholar_results:
            task = asyncio.create_task(
                scholar_result_to_claims(session, scholar_result)
            )
            tasks.append(task)
        i = 0
        claims = []
        for task in asyncio.as_completed(tasks):
            task_claims, title = await task
            claims += task_claims
            i += 1
            set_progress(
                i / len(tasks), f"Extracted claims from '{title}' ({i}/{len(tasks)})"
            )
            set_claims_view(claims)
        return claims


async def async_add_questions_to_claims(claims, question, set_progress):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for claim in claims:
            if not claim.question:
                task = asyncio.create_task(answer_to_question(session, claim, question))
                tasks.append(task)
        i = 0
        for task in asyncio.as_completed(tasks):
            claim, question = await task
            claim.question = question
            i += 1
            set_progress(
                i / len(tasks), f"Computed question '{question}' ({i}/{len(tasks)})"
            )
        return claims


def get_event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # no event loop running:
        loop = asyncio.new_event_loop()
    else:
        nest_asyncio.apply()
    return loop


def scholar_results_to_claims(scholar_results, set_progress, set_claims_view):
    loop = get_event_loop()
    result = loop.run_until_complete(
        async_scholar_results_to_claims(scholar_results, set_progress, set_claims_view)
    )
    return result


def add_questions_to_claims(claims, question, set_progress):
    loop = get_event_loop()
    result = loop.run_until_complete(
        async_add_questions_to_claims(claims, question, set_progress)
    )
    return result


@st.cache(suppress_st_warning=True, persist=True, show_spinner=False, max_entries=30)
def get_scholar_results(query, min_year):
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": serpapi_api_key,
        "num": 20,
        "as_ylo": min_year,
    }
    search = GoogleSearch(params)
    data = search.get_dict()
    return data["organic_results"]


def get_unique_claims(claims):
    unique_claims = []
    seen_claim_texts = set()
    for claim in claims:
        if claim.text in seen_claim_texts:
            continue
        unique_claims.append(claim)
        seen_claim_texts.add(claim.text)
    return unique_claims


def text_to_sentences(text: str):
    """
    Convert text to sentences using spacy
    """
    doc = spacy_nlp(text)
    return [sent.text for sent in doc.sents]


def markdown_list(lines):
    """
    Render list of lines as markdown list
    """
    return "\n".join(f"- {line}" for line in lines)


def get_options():
    with st.expander("Options"):
        col1, col2 = st.columns(2)
        with col1:
            min_year = st.number_input(
                "Year at least", min_value=1950, max_value=2021, value=2011
            )
        with col2:
            min_citations = st.number_input("Citations at least", min_value=0, value=10)
        col3, col4 = st.columns(2)
        with col3:
            require_venue = st.checkbox(
                "Only include publications with venue (journal or conference)",
                value=True,
            )
        with col4:
            min_relevance_score = st.number_input(
                "Relevance at least", min_value=-10.0, value=2.5
            )
    return SimpleNamespace(
        min_year=min_year,
        min_citations=min_citations,
        require_venue=require_venue,
        min_relevance_score=min_relevance_score,
    )


def show_claim(claim, options):
    source = claim.source
    citation_count = source.get("citationCount")
    authors = source.get("authors")
    first_author_name = authors[0]["name"] if authors else "Unknown"
    if len(authors) > 1:
        author_text = f"{first_author_name} et al"
    else:
        author_text = first_author_name
    year = source.get("year")
    venue = source.get("venue")
    st.markdown(
        f"""<span class="authors">{claim.question}</span>
> {claim.text}
""",
        unsafe_allow_html=True,
    )
    with st.expander(f"{citation_count} citations, {year}"):
        st.markdown(
            f"""
[{source.get('title')}]({source.get('url')})

{markdown_list(text_to_sentences(source.get("abstract")))}
- *{citation_count} citations @ {venue}*
- *Relevance: {claim.score:.2f}*
"""
        )


def claim_is_visible(claim, options):
    if claim.score < options.min_relevance_score:
        return False
    source = claim.source
    if source.get("citationCount") < options.min_citations:
        return False
    venue = source.get("venue")
    if (venue == "") and not options.require_venue:
        return False
    return True


def main():

    if "claims" not in st.session_state:
        st.session_state["claims"] = {}

    st.markdown(css.main, unsafe_allow_html=True)

    app_state = st.experimental_get_query_params()
    url_question = app_state.get("q", [""])[0]
    question = st.text_input(
        "Research question",
        value=url_question,
        help="For example: How does creatine affect cognition?",
    )
    st.experimental_set_query_params(q=question)

    options = get_options()

    if not question:
        return

    progress_text = st.empty()
    st.text("")

    progress_text.caption("Retrieving papers...")

    scholar_results = get_scholar_results(question, options.min_year)

    if not scholar_results:
        return

    progress_text.caption("Extracting claims...")

    claims_view = st.empty()

    def set_progress(perc, text):
        progress_text.caption(text)

    def set_claims_view(claims):
        c = claims_view.container()
        with c:
            seen_paper_titles = set()
            unique_claims = get_unique_claims(claims)
            scored_claims = score_claims(question, unique_claims)
            filtered_claims = [
                claim for claim in scored_claims if claim_is_visible(claim, options)
            ]
            claims_with_questions = add_questions_to_claims(
                filtered_claims, question, set_progress
            )
            for claim in claims_with_questions:
                source = claim.source
                if not source["title"] in seen_paper_titles:
                    seen_paper_titles.add(source["title"])
                    show_claim(claim, options)

    claims = scholar_results_to_claims(scholar_results, set_progress, set_claims_view)

    progress_text.empty()

    encoded_question = urllib.parse.quote(question)
    st.markdown(
        f"""
<div align="center">
  <a href="https://www.google.com/search?q={encoded_question}" target="_blank">Google</a> -
  <a href="https://scholar.google.com/scholar?q={encoded_question}">Google Scholar</a> -
  <a href="https://www.semanticscholar.org/search?q={encoded_question}">Semantic Scholar</a>
  </div>
""",
        unsafe_allow_html=True,
    )


main()
