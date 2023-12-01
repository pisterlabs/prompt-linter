import openai

from typing import Any, Tuple

import utils.clients.openai_config
from utils.clients import MetaphorClient


def search_for_results(
    query: str = None,
    num_results: int = 10,
    include_domains: list = None,
    exclude_domains: list = None,
    use_autoprompt: bool = True,
) -> Tuple[Any, Any]:
    client_instance = MetaphorClient()
    client = client_instance.client

    rephrased_query = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that helps reformat sentences.",
            },
            {
                "role": "user",
                "content": f"Following this example: instead of searching for `What's the best way to get started with cooking?`, try rephrase the question to resemble an answer; `This is the best tutorial on how to get started with cooking`: Rephrase the following question into an answer with a colon at end.:\n{query}",
            },
        ],
    )

    search_query = rephrased_query["choices"][0]["message"]["content"]

    res = client.search(
        query=search_query,
        num_results=num_results,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        use_autoprompt=use_autoprompt,
    )

    content_ids = [result.id for result in res.results]
    content_response = client.get_contents(content_ids)

    result_content_map = {content.id: content for content in content_response.contents}

    for result in res.results:
        content = result_content_map.get(result.id)
        if content:
            result.extract = content.extract

    return res.results, content_response.contents
