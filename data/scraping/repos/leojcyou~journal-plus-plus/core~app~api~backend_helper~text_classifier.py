import cohere

# Biq6EEDJ1nEmcC9nEv4kqkX7gPGzYwerONOrorAq

co = cohere.Client('8wg6SXwL4hlWTQ3hz2yR7U36pCf9Y6ysjgOCuNRD') # This is your trial API key

def get_response(text: str):
    response = co.generate(
    model='command',
    prompt=f'I have these 6 categories: \"Career\", \"Academics\", \"Interpersonal Relationships\", and \"Personal Development\".\n Classify this sentence by outputting only one category name, nothing else: \"{text}\"?',
    max_tokens=2,
    temperature=0.9,
    k=0,
    stop_sequences=[],
    return_likelihoods='NONE')
    return response.generations[0].text

def getAllClassifications(sentences: list):
    classifications = []
    
    for sentence in sentences:
        resp = get_response(sentence)
        classifications.append(resp[1:])

    return classifications