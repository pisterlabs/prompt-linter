from dataclasses import dataclass

from fastapi import Depends, HTTPException, status, Request

from modal import Stub, Secret, web_endpoint, Image

stub = Stub()

custom_image = Image.debian_slim().pip_install(
    "pygithub", "pymongo", "cohere", "jinja2"
)

GITHUB_APP_ID = 381420


@stub.function(secret=Secret.from_name("backseat"), image=custom_image)
def draft_issue_response(repo_name: str, issue_number: int):
    import os
    import cohere
    from pymongo import MongoClient
    from github import Github, GithubIntegration

    cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
    mongo_client = MongoClient(os.getenv("MONGODB_URI"))

    embeddings_collection = mongo_client["backseat"]["embeddings"]

    app = GithubIntegration(GITHUB_APP_ID, os.getenv("GITHUB_APP_PRIVATE_KEY"))
    print("Got integration")

    owner = repo_name.split("/")[0]
    repo = repo_name.split("/")[1]
    print(f"Owner: {owner}, repo: {repo}")
    installation = app.get_installation(owner, repo)
    installation_token = app.get_access_token(installation.id).token

    print(f"Installation token: {installation_token}")

    github = Github(installation_token)

    # list issues
    gh_repo = github.get_repo(repo_name)
    issue = gh_repo.get_issue(issue_number)
    repo_id = gh_repo.id

    print("Found issue")

    mongo_client["backseat"]["issues"].update_one(
        {
            "type": "issue",
            "repoId": repo_id,
            "issueNumber": issue_number,
        },
        {"$set": {"status": "generating"}},
    )

    issue_content = f"{issue.title}\n\n{issue.body}"

    for comment in issue.get_comments():
        issue_content += f"\n\n{comment.body}"

    from pprint import pprint

    pprint(issue)
    print(issue_content)

    # get the issue's embeddings
    cohere_response = cohere_client.embed(
        texts=[issue_content],
        model="small",
    )

    embedding = cohere_response.embeddings[0]

    contents = embeddings_collection.aggregate(
        [
            {
                "$search": {
                    "index": "embeddings",
                    "knnBeta": {
                        "vector": embedding,
                        "path": "cohereSmallEmbedding",
                        "k": 11,
                        "filter": {
                            "compound": {
                                "mustNot": {
                                    "equals": {
                                        "value": issue_number,
                                        "path": "issueNumber",
                                    },
                                },
                            }
                        },
                    },
                    "scoreDetails": True,
                },
            },
            {
                "$project": {
                    "score": {
                        "$meta": "searchScoreDetails",
                    },
                    "type": 1,
                    "path": 1,
                    "issueNumber": 1,
                    "issueType": 1,
                    "repoId": 1,
                    "text": 1,
                }
            },
        ]
    )

    print("Got similar issues")

    # generate the text for relevant issues
    relevant_content = ""

    for content in contents:
        if content["score"]["value"] < 0.75:
            print("Not using content")
            print(content)
            continue

        relevant_content += "\n" + "-" * 80 + "\n"
        relevant_content += f"{content['type']} "
        if content["type"] == "issue" or content["type"] == "pr":
            relevant_content += f"#{content['issueNumber']}"
        elif content["type"] == "file":
            relevant_content += f"{content['path']}"

        # get the issue text
        relevant_content += f"\n{content['text']}\n\n"
        relevant_content += "\n" + "-" * 80 + "\n"

    import jinja2

    # read the prompt from draft_response_prompt.jinja
    template = jinja2.Template(
        """\
You are an AI assistant responsible for helping users triage issues in open-source projects.

A user has just opened this issue:
```\n{{ issue_content }}\n```

Draft a response (and only a response) that will be written as if you were the project administrator. Your response should be based on the following (in order):
- if you can propose code changes to fix a bug or implement a feature, do so
- if the issue is a duplicate, say so and encourage the user to close the issue and comment on the original issue
- if the issue is a feature request, say so and encourage the user to open a pull request
- if the issue is a bug, say so and encourage the user to open a pull request
- if the issue does not have enough information, ask the user to provide more information

Your tone should be very gracious and helpful. You should not be sarcastic or rude. You should not be overly formal.

You should not include the instructions in your response. Try to keep the response as short as possible.

Do not include text like "If you are requesting a new feature, ...". Instead, you should be able to determine if the issue is a feature request or a bug and respond accordingly.

We've done a search to find potentially relevant content in the repository, and found these:
{{ relevant_content }}
"""
    )

    # render the prompt
    prompt = template.render(
        issue_content=issue_content,
        relevant_content=relevant_content,
    )

    print("=" * 80)
    print(prompt)
    print("=" * 80)

    # generate the response
    response = cohere_client.generate(
        prompt=prompt,
        model="command",
        max_tokens=200,
        temperature=0.1,
        k=200,
    )

    first_response = response[0].text

    print("Response:")
    print(first_response)
    pprint(
        {
            "type": "issue",
            "repoId": repo_id,
            "issueNumber": issue_number,
        }
    )

    # update the issue with the similar issues
    mongo_client["backseat"]["issues"].update_one(
        {
            "type": "issue",
            "repoId": repo_id,
            "issueNumber": issue_number,
        },
        {
            "$set": {
                "draftResponse": first_response,
                "status": "done",
            },
        },
    )
