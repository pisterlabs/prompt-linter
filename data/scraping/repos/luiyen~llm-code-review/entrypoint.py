#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import json
import os
from typing import List

import click
import requests
from langchain import HuggingFaceHub, LLMChain, PromptTemplate
from loguru import logger


def check_required_env_vars():
    """Check required environment variables"""
    required_env_vars = [
        "API_KEY",
        "GITHUB_TOKEN",
        "GITHUB_REPOSITORY",
        "GITHUB_PULL_REQUEST_NUMBER",
        "GIT_COMMIT_HASH",
    ]
    for required_env_var in required_env_vars:
        if os.getenv(required_env_var) is None:
            raise ValueError(f"{required_env_var} is not set")


def create_a_comment_to_pull_request(
        github_token: str,
        github_repository: str,
        pull_request_number: int,
        git_commit_hash: str,
        body: str):
    """Create a comment to a pull request"""
    headers = {
        "Accept": "application/vnd.github.v3.patch",
        "authorization": f"Bearer {github_token}"
    }
    data = {
        "body": body,
        "commit_id": git_commit_hash,
        "event": "COMMENT"
    }
    url = f"https://api.github.com/repos/{github_repository}/pulls/{pull_request_number}/reviews"
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response


def chunk_string(input_string: str, chunk_size) -> List[str]:
    """Chunk a string"""
    chunked_inputs = []
    for i in range(0, len(input_string), chunk_size):
        chunked_inputs.append(input_string[i:i + chunk_size])
    return chunked_inputs


def get_review(
        repo_id: str,
        diff: str,
        temperature: float,
        max_new_tokens: int,
        top_p: float,
        top_k: int,
        prompt_chunk_size: int
):
    """Get a review"""
    # Chunk the prompt
    chunked_diff_list = chunk_string(input_string=diff, chunk_size=prompt_chunk_size)
    # Get summary by chunk
    chunked_reviews = []
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": temperature,
                      "max_new_tokens": max_new_tokens,
                      "top_p": top_p,
                      "top_k": top_k},
                      huggingfacehub_api_token=os.getenv("API_KEY")
    )
    for chunked_diff in chunked_diff_list:
        question=chunked_diff
        template = """Provide a concise summary of the bug found in the code, describing its characteristics, 
        location, and potential effects on the overall functionality and performance of the application.
        Present the potential issues and errors first, following by the most important findings, in your summary
        Important: Include block of code / diff in the summary also the line number.

        Diff:

        {question}
        """

        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        review_result = llm_chain.run(question)
        chunked_reviews.append(review_result)

    # If the chunked reviews are only one, return it
    if len(chunked_reviews) == 1:
        return chunked_reviews, chunked_reviews[0]

    question="\n".join(chunked_reviews)
    template = """Summarize the following file changed in a pull request submitted by a developer on GitHub,
    focusing on major modifications, additions, deletions, and any significant updates within the files.
    Do not include the file name in the summary and list the summary with bullet points.
    Important: Include block of code / diff in the summary also the line number.
    
    Diff:
    {question}
    """
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    summarized_review = llm_chain.run(question)
    return chunked_reviews, summarized_review


def format_review_comment(summarized_review: str, chunked_reviews: List[str]) -> str:
    """Format reviews"""
    if len(chunked_reviews) == 1:
        return summarized_review
    unioned_reviews = "\n".join(chunked_reviews)
    review = f"""<details>
    <summary>{summarized_review}</summary>
    {unioned_reviews}
    </details>
    """
    return review


@click.command()
@click.option("--diff", type=click.STRING, required=True, help="Pull request diff")
@click.option("--diff-chunk-size", type=click.INT, required=False, default=3500, help="Pull request diff")
@click.option("--repo-id", type=click.STRING, required=False, default="gpt-3.5-turbo", help="Model")
@click.option("--temperature", type=click.FLOAT, required=False, default=0.1, help="Temperature")
@click.option("--max-new-tokens", type=click.INT, required=False, default=250, help="Max tokens")
@click.option("--top-p", type=click.FLOAT, required=False, default=1.0, help="Top N")
@click.option("--top-k", type=click.INT, required=False, default=1.0, help="Top T")
@click.option("--log-level", type=click.STRING, required=False, default="INFO", help="Presence penalty")
def main(
        diff: str,
        diff_chunk_size: int,
        repo_id: str,
        temperature: float,
        max_new_tokens: int,
        top_p: float,
        top_k: int,
        log_level: str
):
    # Set log level
    logger.level(log_level)
    # Check if necessary environment variables are set or not
    check_required_env_vars()

    # Request a code review
    chunked_reviews, summarized_review = get_review(
        diff=diff,
        repo_id=repo_id,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        prompt_chunk_size=diff_chunk_size
    )
    logger.debug(f"Summarized review: {summarized_review}")
    logger.debug(f"Chunked reviews: {chunked_reviews}")

    # Format reviews
    review_comment = format_review_comment(summarized_review=summarized_review,
                                           chunked_reviews=chunked_reviews)
    # Create a comment to a pull request
    create_a_comment_to_pull_request(
        github_token=os.getenv("GITHUB_TOKEN"),
        github_repository=os.getenv("GITHUB_REPOSITORY"),
        pull_request_number=int(os.getenv("GITHUB_PULL_REQUEST_NUMBER")),
        git_commit_hash=os.getenv("GIT_COMMIT_HASH"),
        body=review_comment
    )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
