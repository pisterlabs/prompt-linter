import uvicorn
import logging
import json
import asyncio
import httpx
import re

from collections import Counter
from transformers import pipeline
from dotenv import dotenv_values
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from anthropic import AsyncAnthropic, HUMAN_PROMPT, AI_PROMPT
from refined.inference.processor import Refined
from random import random

from transformers import pipeline

from prompts import (
    YES_NO_Q_PROMPT,
    SUMMARISATION_PROMPT,
    MULTIPLE_ANSWERS_PROMPT,
    NUMERIC_ANSWER_PROMPT,
    OPINIONS_PROMPT,
    TOT_PROMPT,
)

config = dotenv_values()
anthropic = AsyncAnthropic(api_key=config["CLAUDE_API_KEY"], max_retries=5, timeout=60)
app = FastAPI()
app.allowed_claude_connections = 2

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn.info")

classifier = pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli", device="mps"
)

refined = Refined.from_pretrained(
    model_name="wikipedia_model_with_numbers", entity_set="wikipedia"
)


class Query(BaseModel):
    text: str


async def ask_claude(prompt):
    logger.info("Asking Claude..")
    completion = await anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=1000,
        prompt=prompt,
    )
    return completion.completion


async def news_snippets_for_query(client, query, offset=0):
    logger.info(f"Retrieving relevant snippets (page={offset + 1})...")
    url = f"https://api.ydc-index.io/news?q={query}"
    headers = {"X-API-Key": config["YOU_API_KEY"]}
    params = {"query": query, "count": 20, "offset": offset}
    resp = await client.get(url, params=params, headers=headers)
    return resp.json()["news"]["results"]


async def rag_snippets_for_query(client, query, offset=0):
    logger.info(f"Retrieving relevant snippets (page={offset + 1})...")
    url = f"https://api.ydc-index.io/rag?q={query}"
    headers = {"X-API-Key": config["YOU_API_KEY"]}
    params = {"query": query, "count": 20, "offset": offset}
    resp = await client.get(url, params=params, headers=headers)
    return resp.json()


def classifty_query(text):
    candidate_labels = [
        "yes or no question",
        "question with more than 2 possible answers",
        "question requiring a numeric answer",
        "not a question",
    ]
    output = classifier(text, candidate_labels, multi_label=False)
    labels_with_scores = sorted(
        zip(output["labels"], output["scores"]), key=lambda t: t[1], reverse=True
    )
    pred = labels_with_scores[0][0]

    q = text.lower()
    if q.startswith("is "):
        pred = "yes or no question"
    elif q.startswith("how many "):
        pred = "question requiring a numeric answer"

    logger.info(f"Classification prediction: {pred}")
    return pred


def route_to_relevant_prompt(pred, query, context):
    if pred == "yes or no question":
        logger.info(f"Processing Yes/No question")
        prompt = YES_NO_Q_PROMPT.format(
            human_prompt=HUMAN_PROMPT, query=query, context=context, ai_prompt=AI_PROMPT
        )
    elif pred == "question with more than 2 possible answers":
        logger.info(f"Processing a question with more than 2 possible answers")
        prompt = MULTIPLE_ANSWERS_PROMPT.format(
            human_prompt=HUMAN_PROMPT,
            query=query,
            context=context,
            ai_prompt=AI_PROMPT,
        )
    elif pred == "question requiring a numeric answer":
        logger.info(f"Processing a question with numeric answer")
        prompt = NUMERIC_ANSWER_PROMPT.format(
            human_prompt=HUMAN_PROMPT, query=query, context=context, ai_prompt=AI_PROMPT
        )

    return prompt + "{"


@app.get("/api/cache")
async def get_cached_queries():
    with open("store.json", "r") as fr:
        store = json.load(fr)
    logger.info(f"Returning {len(store)} cached queries..")
    return list(store.keys())


@app.post("/api/news")
async def get_news(query: Query):
    logger.info(f"Received question: {query.text}")
    logger.info(f"Searching for relevant news..")

    query = query.text

    if len(query.split(" ")) <= 3:
        return []

    with open("store.json", "r") as fr:
        store = json.load(fr)

    if store.get(query) and store[query].get("results"):
        logger.info(f"Returning cached news results")
        return store[query]["results"]

    async with httpx.AsyncClient() as client:
        futures = [
            news_snippets_for_query(client, query),
            news_snippets_for_query(client, query, offset=1),
            news_snippets_for_query(client, query, offset=2),
        ]
        results = await asyncio.gather(*futures)

    # Getting news search results
    search_results = [record for page in results for record in page]
    logger.info(f"Got {len(search_results)} news results")
    store[query] = {"results": search_results}

    with open("store.json", "w") as fw:
        logger.info(f"Saved news response to store")
        json.dump(store, fw)

    return search_results


def postprocess_response(text):
    if "Here is a summary" in text or "Here is a paragraph":
        return text.split(":")[1].strip()
    return text


def prepare_response(pred, results):
    response = {}
    summary = postprocess_response(results[1])

    if pred == "yes or no question":
        logger.info(f"Yes/No question answer: {results[0]}")
        answer = json.loads("{" + results[0])
        num_considered = sum(answer.values())
        response = {
            "type_of_question": "yes_or_no",
            "answer": {k: round(v / num_considered, 4) for k, v in answer.items()},
            "summary": summary,
            "num_considered": num_considered,
        }
    elif pred == "question with more than 2 possible answers":
        logger.info(f"Multiple answers question: {results[0]}")
        statements = json.loads("{" + results[0])["statements"]
        response = {
            "type_of_question": "multiple_answers",
            "answer": statements,
            "summary": summary,
        }
    elif pred == "question requiring a numeric answer":
        logger.info(f"Numeric answer question: {results[0]}")
        answer = json.loads("{" + results[0])

        response = {
            "type_of_question": "numeric_answer",
            "answer": answer,
            "summary": summary,
        }

    return response


@app.post("/api/opinions")
async def opinions(query: Query):
    logger.info(("Getting opinions"))
    query = query.text
    if len(query.split(" ")) <= 3:
        return {}

    with open("store.json", "r") as fr:
        store = json.load(fr)

    if store.get(query) and store[query].get("opinions"):
        logger.info(f"Returning cached opinions")
        return store[query]["opinions"]

    search_results = store[query]["results"]
    context = " ".join(
        [f"{r['description']} (published {r['age']})" for r in search_results]
    )
    opinions_prompt = OPINIONS_PROMPT.format(
        human_prompt=HUMAN_PROMPT,
        query=query,
        context=context,
        ai_prompt=AI_PROMPT,
    )

    experts_prompt = TOT_PROMPT.format(
        human_prompt=HUMAN_PROMPT,
        query=query,
        context=context,
        ai_prompt=AI_PROMPT,
    )

    while app.allowed_claude_connections < 2:
        await asyncio.sleep(5)

    app.allowed_claude_connections -= 2
    futures = [
        ask_claude(opinions_prompt),
        ask_claude(experts_prompt),
    ]
    async with asyncio.timeout(60):
        results = await asyncio.gather(*futures)

    app.allowed_claude_connections += 2
    takes, experts = results
    logger.info(f"Takes {takes}")
    logger.info(f"Experts {experts}")
    debate = re.findall(
        r"<(optimist|skeptic)>([\s\S]*?)</(optimist|skeptic)>", takes.replace("\n", "")
    )
    debate = [{"side": t[0], "text": t[1]} for t in debate]
    conclusion = re.search(
        r"<conclusion>([\s\S]*?)</conclusion>", takes.replace("\n", "")
    ).group(1)

    takes = {"debate": debate, "conclusion": conclusion}
    expert_analysis = json.loads(
        re.search(r"<answer>([(\s\S]*?)</answer>", experts.replace("\n", "")).group(1)
    )

    response = {"takes": takes, "expert_analysis": expert_analysis}
    store[query]["opinions"] = response
    with open("store.json", "w") as fw:
        logger.info(f"Saved search opinions to store")
        json.dump(store, fw)

    return response


@app.post("/api/entities")
def entities_and_sentiment(query: Query):
    logger.info("Working on entities & sentiment")
    query = query.text

    if len(query.split(" ")) <= 3:
        return {}

    with open("store.json", "r") as fr:
        store = json.load(fr)

    if store.get(query) and store[query].get("entities"):
        logger.info(f"Returning cached entities")
        return store[query]["entities"]

    search_results = store[query]["results"]
    text = ". ".join([r["description"] for r in search_results])
    logger.info("Getting entities")
    spans = refined.process_text(text)

    common_entities = []
    allowed_types = {"ORG", "PERSON", "FAC", "GPE", "EVENT", "WORK_OF_ART", "LOC"}

    for span in spans:
        key = f"{span.text}__{span.coarse_mention_type}"
        if span.coarse_mention_type in allowed_types:
            common_entities.append(key)

    top_k = 5
    key_entities_with_counts = dict(Counter(common_entities).most_common(top_k))
    logger.info(key_entities_with_counts)

    label_templates = [
        "positive sentiment towards {}",
        "negative sentiment towards {}",
        "no sentiment expressed towards {}",
    ]

    key_entities_with_sentiments = {}
    clusters = {}
    cluster_ents_with_counts = {}
    entity_types = {}

    for k in key_entities_with_counts.keys():
        ent = k.split("__")[0]
        clusters[ent] = []
        cluster_ents_with_counts[ent] = 0
        for k2 in key_entities_with_counts.keys():
            ent2 = k2.split("__")[0]
            if ent != ent2 and ent in ent2 or ent2 in ent:
                clusters[ent].append(ent2)
                cluster_ents_with_counts[ent] += key_entities_with_counts[k2]

    seen = set()
    key_uniq_entities_with_counts = {}

    for k, v in key_entities_with_counts.items():
        ent_name, ent_type = k.split("__")
        if ent_name in seen:
            continue
        key_uniq_entities_with_counts[ent_name] = cluster_ents_with_counts[ent_name]
        entity_types[ent_name] = ent_type
        seen.add(ent_name)
        for c in clusters[ent_name]:
            seen.add(c)

    top_key_uniq_entities_with_counts = [
        (k, v) for k, v in key_uniq_entities_with_counts.items()
    ][:top_k]
    logger.info("Getting sentiment for mentions")

    for ent, count in top_key_uniq_entities_with_counts:
        labels = [c.format(ent) for c in label_templates]
        ent_inputs = []
        ent = ent.split(".")[0]
        p = 10 / count

        for result in search_results:
            if ent in result["description"] and random() <= p:
                ent_inputs.append(result["description"])

        outputs = classifier(ent_inputs, labels, multi_label=False)
        for output in outputs:
            labels_with_scores = sorted(
                zip(output["labels"], output["scores"]),
                key=lambda t: t[1],
                reverse=True,
            )
            pred = labels_with_scores[0][0]
            pred = pred.split(" ")[0]
            key_entities_with_sentiments[ent] = key_entities_with_sentiments.get(
                ent, []
            ) + [pred]

    key_ents_with_sentiment = {}

    for k, v in key_entities_with_sentiments.items():
        d = dict(Counter(v))
        s = {
            "positive": d.get("positive", 0),
            "negative": d.get("negative", 0),
            "neutral": d.get("neutral", d.get("no", 0)),
        }
        total = sum(s.values())
        s["pos_perc"] = s["positive"] / total
        s["neg_perc"] = s["negative"] / total
        s["neu_perc"] = s["neutral"] / total
        key_ents_with_sentiment[k] = s

    response = {
        "entities": dict(top_key_uniq_entities_with_counts),
        "sentiment": key_ents_with_sentiment,
        "entity_types": entity_types,
    }
    store[query]["entities"] = response
    with open("store.json", "w") as fw:
        logger.info(f"Saved entities to store")
        json.dump(store, fw)

    logger.info("Returning entities & sentiment")
    return response


@app.post("/api/search")
async def search(query: Query):
    logger.info(f"Received question: {query.text}")

    query = query.text

    if len(query.split(" ")) <= 3:
        return {
            "type_of_question": "not_a_question",
            "answer": {},
            "summary": "",
        }

    with open("store.json", "r") as fr:
        store = json.load(fr)

    if store.get(query) and store[query].get("response"):
        logger.info(f"Returning cached response")
        return store[query]["response"]

    search_results = store[query]["results"]
    context = " ".join(
        [f"{r['description']} (published {r['age']})" for r in search_results]
    )
    logger.info(f"Using {len(search_results)} results")
    summary_prompt = SUMMARISATION_PROMPT.format(
        human_prompt=HUMAN_PROMPT, context=context, ai_prompt=AI_PROMPT
    )

    pred = classifty_query(query)
    prompt = route_to_relevant_prompt(pred, query, context)

    while app.allowed_claude_connections < 2:
        await asyncio.sleep(5)
    app.allowed_claude_connections -= 2

    futures = [
        ask_claude(prompt),
        ask_claude(summary_prompt),
    ]
    async with asyncio.timeout(60):
        results = await asyncio.gather(*futures)

    app.allowed_claude_connections += 2
    response = prepare_response(pred, results)

    store[query]["response"] = response
    with open("store.json", "w") as fw:
        logger.info(f"Saved search response to store")
        json.dump(store, fw)

    return response


if __name__ == "__main__":
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"][
        "fmt"
    ] = "%(asctime)s %(levelname)s %(module)s.%(funcName)s: %(message)s"

    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=log_config)
