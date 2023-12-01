import openai


def call_openai(word):
    # function calls openai and extract the classification for a given word
    openai.api_key = "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"

    in_prompt = ("The following is a list of keywords an the categories they fall into\n\n"
                 "result: phenomenon, statement, ending, prove, produce, happen \n"
                 "carbon: chemical element, paper, copy\n"
                 "conversion: transformation, calculation, score, redemption, change, exchange, change\n"
                 "reactor: electrical device, apparatus\n"
                 "co2: dioxide, greenhouse gas\n"
                 "hydrogen: chemical element, gas\n"
                 "support: activity, aid, influence, operation, validation, resource, supporting structure, activity\n"
                 "activity: act, state, organic process, capability, process, trait\n"
                 "reaction: chemical process, idea, bodily process, force, response, conservatism, resistance\n"
                 "energy: physical phenomenon, force, drive, liveliness, good health, executive department\n"
                 + word + ":")

    response = openai.Completion.create(
        # davinci engine is most capable model
        engine="davinci",
        # creating an example for API
        prompt=in_prompt,
        temperature=0,
        max_tokens=100,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=['\n']
        )
    hypernyms = response["choices"][0]["text"]
    return hypernyms
