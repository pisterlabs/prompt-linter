from dataclasses import dataclass

from fastapi import Depends, HTTPException, status, Request

from modal import Stub, Secret, web_endpoint, Image

stub = Stub()

custom_image = Image.debian_slim().pip_install("pygithub", "pymongo", "cohere")


@stub.function(secret=Secret.from_name("backseat"), image=custom_image)
def check_for_dupes(repo_id: int, issue_number: int, verbose=False):
    import os
    import cohere
    from pymongo import MongoClient

    cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
    mongo_client = MongoClient(os.getenv("MONGODB_URI"))

    embeddings_collection = mongo_client["backseat"]["embeddings"]

    mongo_client["backseat"]["issues"].update_one(
        {
            "type": "issue",
            "repoId": repo_id,
            "issueNumber": issue_number,
        },
        {"$set": {"status": "generating"}},
    )

    # first, get the embeddings for the new issue
    issue = embeddings_collection.find_one(
        {
            "type": "issue",
            "repoId": repo_id,
            "issueNumber": issue_number,
        }
    )

    similar_issues = embeddings_collection.aggregate(
        [
            {
                "$search": {
                    "index": "embeddings",
                    "knnBeta": {
                        "vector": issue["cohereSmallEmbedding"],
                        "path": "cohereSmallEmbedding",
                        "k": 11,
                        "filter": {
                            "compound": {
                                "must": {
                                    "text": {
                                        "query": "issue",
                                        "path": "type",
                                    }
                                },
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
                    "issueNumber": 1,
                    "issueType": 1,
                    "repoId": 1,
                }
            },
        ]
    )

    from pprint import pprint

    mdb_similar_issues = []
    for similar_issue in similar_issues:
        mdb_similar_issues.append(
            {
                "issueNumber": similar_issue["issueNumber"],
                "issueType": "issue",
                "repoId": similar_issue["repoId"],
                "score": similar_issue["score"]["value"],
            }
        )
        if(verbose):
            pprint(similar_issue)

    issues_collection = mongo_client["backseat"]["issues"]

    update = {
        "$set": {
            "similarIssues": mdb_similar_issues,
            "status": "done",
        },
    }
    if(verbose):
        print("updating...")
        pprint(update)

    # update the issue with the similar issues
    issues_collection.update_one(
        {
            "type": "issue",
            "repoId": repo_id,
            "issueNumber": issue_number,
        },
        update,
    )

    return {
        "issue": issue,
        "similarIssues": mdb_similar_issues,
    }
