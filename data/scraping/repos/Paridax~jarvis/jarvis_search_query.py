import openai

prompt_extension = """"search_query" (necessary fields; query: string)"""


def search_query(dictionary, settings):
    # load openai api key
    openai.api_key = settings["openai_key"]

    query = dictionary.get("query")
    if query is None:
        return False
    if settings["debug"]:
        print("Searching for: ", query)
    else:
        print("Searching the web...")

    search_results = settings["google_search"].search(query, text=True, links=5)

    prompt = f"""Answer the query given using the text given, only give the direct answer, do not repeat the question or give background or extra information. If the prompt asks for a link make sure to return one, the text is from a google search of the query, at the end of the text will be links and their corresponding text. Answer this question with the data provided: \"{query}\". Here is the search data: {search_results}"""

    response = openai.ChatCompletion.create(
        messages=[
            {"role": "user", "content": prompt},
        ],
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=1000,
    )

    answer = response["choices"][0]["message"]["content"]

    # print price of prompt and response in usd
    if settings["debug"]:
        print(f"Total cost in dollars: ${response['usage']['total_tokens'] * 0.000002}")

    if settings["debug"]:
        print(f"ANSWER: {answer}")
    else:
        print(answer)
