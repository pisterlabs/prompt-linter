def ask(query):
    import openai
    import json

    with open("./secretes.json", "r") as f:
        json_file = json.load(f)
    openai.organization = "org-56LTi4oJG3ERd1cSlyN6sv2t"
    openai.api_key = json_file['api_key']

    # print(query)

    response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=query,
    temperature=0.5,
    max_tokens=1000,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    return response['choices'][0]['text']