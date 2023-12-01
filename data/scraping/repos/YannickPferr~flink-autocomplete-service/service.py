import openai
import secrets_config
import config
import json
import heapq

openai.api_key = secrets_config.openapikey


def autocomplete_from_docs(query: str) -> list:
    # load code snippets from file
    code_snippets_dict = {}
    with open('snippets.json') as f:
        code_snippets_dict = json.load(f)

    # build a max heap that contains the best suggestions ranked by earliest occurence of query
    max_heap = []
    for page in code_snippets_dict:
        code_snippets_from_page = code_snippets_dict[page]
        for code_snippet in code_snippets_from_page:
            try:
                index_of_query = code_snippet.lower().index(query.lower())
                # if heap has still space add snippet directly
                if len(max_heap) < config.autocomplete_max_suggestions:
                    heapq.heappush(max_heap, (-index_of_query, code_snippet))
                    continue

                # otherwise check if this element has a better score than the worst element in the heap
                if index_of_query < max_heap[0][0]:
                    # remove snippet and add better snippet
                    heapq.heappop(max_heap)
                    heapq.heappush(max_heap, (-index_of_query, code_snippet))
            except ValueError:
                pass

    # iterate heap from the back (last element is best)
    suggestions = list()
    while len(max_heap) > 0:
        suggestions.insert(0, heapq.heappop(max_heap)[1])

    return suggestions


def autocomplete_with_gpt(query: str) -> list:
    url = "https://api.openai.com/v1/engines/davinci/completions"

    response = openai.Completion.create(
        model=config.gpt_model,
        prompt=f"{query}\n\n/* Generate multiple auto completions for the previous Flink SQL */",
        temperature=0.7,
        max_tokens=config.gpt_max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Response
    text_response = response.choices[0].text

    # text_response = "\n\nSELECT order_id, item_name FROM orders WHERE\nSELECT order_id, customer_name FROM orders WHERE\nSELECT order_id, quantity FROM orders WHERE\nSELECT order_id, shipping_address FROM orders WHERE\nSELECT order_id, order_date FROM orders WHERE"

    # Removing empty strings and break line
    suggestions = list(filter(None, text_response.split("\n")))

    return suggestions


def explanation_with_gpt(query: str) -> str:
    response = openai.Completion.create(
        model=config.gpt_model,
        prompt=f"{query}\n\n/* Describe the previous Flink SQL */",
        temperature=0.7,
        max_tokens=config.gpt_max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].text
