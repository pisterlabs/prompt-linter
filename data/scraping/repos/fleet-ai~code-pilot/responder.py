# pylint: disable=unused-argument
import os
import json
import uuid
import asyncio
import requests

import pinecone
from openai import OpenAI
from fastapi import APIRouter, Request, Response

from auth import get_token
from utils.utils import batch
from utils.chunk import chunk_nltk
from utils.embed import embed_chunks
from utils.query import query_index, parse_results
from constants import (
    MODEL,
    PROMPT,
    EMBEDDINGS_MODEL,
    MAX_CONTEXT_LENGTH_EMBEDDINGS,
    INDEX_NAME,
    INDEX_ENVIRONMENT,
    NAMESPACE,
    BOT_NAME,
)
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pinecone.init(api_key=PINECONE_API_KEY, environment=INDEX_ENVIRONMENT)
index = pinecone.Index(INDEX_NAME)


async def embed_issues(issues):
    vectors = []
    for issue in issues:
        chunks = chunk_nltk(issue["body"])
        embeddings = embed_chunks(
            chunks,
            model=EMBEDDINGS_MODEL,
            token_limit=MAX_CONTEXT_LENGTH_EMBEDDINGS,
        )
        for chunk, embedding in zip(chunks, embeddings):
            metadata = {
                "type": "issue",
                "issue_id": issue["id"],
                "url": issue["html_url"],
                "title": issue["title"],
                "body": chunk,
                "state": issue["state"],
                "labels": issue["labels"],
            }
            vectors.append(
                {
                    "id": str(uuid.uuid4()),
                    "values": embedding,
                    "metadata": metadata,
                }
            )

    for vec_batch in batch(vectors, 100):
        index.upsert(vectors=vec_batch, namespace=NAMESPACE)

    print("Finished embedding issue(s).")


async def respond_to_opened_issue(
    query,
    repo_name,
    repo_url,
    issue_number,
    github_auth_token,
    original_issue=None,
):
    # Make a call to Pinecone to get the top 10 contexts
    docs_issues_results = query_index(
        query=query,
        k=10,
        filter_dict={"type": {"$in": ["documentation", "issue"]}},
        namespace=NAMESPACE,
    )
    code_results = query_index(
        query=query,
        k=5,
        filter_dict={"type": "code"},
        namespace=NAMESPACE,
    )

    # Consolidate the contexts
    results = (
        docs_issues_results + code_results if code_results else docs_issues_results
    )
    context_text = parse_results(repo_url, results)
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": context_text},
        {"role": "user", "content": query},
    ]

    # Add original issue if needs adding
    if original_issue:
        messages.insert(1, {"role": "user", "content": original_issue})

    # Create the OpenAI response
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    # Post a comment
    url = f"https://api.github.com/repos/{repo_name}/issues/{issue_number}/comments"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_auth_token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    body = {
        "body": response.choices[0].message.content
        + f"\n\n*🤖 **Beep boop!** This response was generated using [Fleet Context](https://fleet.so/context)'s library embeddings. It is not meant to be a precise solution, but rather a starting point for your own research.*\n\n*You can ask a follow-up by tagging me @{BOT_NAME}.*"
    }
    response = requests.post(url, headers=headers, data=json.dumps(body), timeout=120)

    # Check the response
    if response.status_code == 201:
        print("Comment posted successfully.")
    else:
        print(f"Failed to post comment. Status code: {response.status_code}")


@router.post("/", response_model=str)
async def github_response(request: Request):
    data = await request.json()

    # Validate JWT and create an auth token
    github_auth_token = get_token(data["installation"]["id"])

    # Entered auth
    if data["action"] == "created" and "account" in data["installation"]:
        print("App installed")
        # Then, loop through all issues and embed
        for repo in data.get("repositories", []):
            repo_name = repo["full_name"]
            issues = []

            page = 1
            url = f"https://api.github.com/repos/{repo_name}/issues?state=all&per_page=100&page={page}"
            headers = {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {github_auth_token}",
                "X-GitHub-Api-Version": "2022-11-28",
            }

            response = requests.get(url, headers=headers, timeout=120).json()
            issues.extend(response)
            # Loop through all responses, paginated
            while len(response) == 100:
                page += 1
                url = f"https://api.github.com/repos/{repo_name}/issues?state=all&per_page=100&page={page}"
                response = requests.get(url, headers=headers, timeout=120).json()
                issues.extend(response)

            asyncio.create_task(embed_issues(issues))

    # Opened issue
    elif data["action"] == "opened":
        print("Issue opened")
        # Respond to opened issue
        asyncio.create_task(
            respond_to_opened_issue(
                query=data["issue"]["body"],
                repo_name=data["repository"]["full_name"],
                repo_url=data["repository"]["html_url"],
                issue_number=data["issue"]["number"],
                github_auth_token=github_auth_token,
            )
        )
        # Embed the issue
        asyncio.create_task(embed_issues([data["issue"]]))

    # Commented on issue
    elif data["action"] == "created" and "issue" in data and "comment" in data:
        if (
            not data["comment"]["user"]["login"] == "issues-responder[bot]"
            and "BOT" not in data["comment"]["user"]["node_id"]
            and f"@{BOT_NAME}" in data["comment"]["body"]
        ):
            print("Commented on issue")
            # Respond to the opened issue
            asyncio.create_task(
                respond_to_opened_issue(
                    query=data["comment"]["body"],
                    repo_name=data["repository"]["full_name"],
                    repo_url=data["repository"]["html_url"],
                    issue_number=data["issue"]["number"],
                    github_auth_token=github_auth_token,
                    original_issue=data["issue"]["body"],
                )
            )

    return Response("", status_code=200)
