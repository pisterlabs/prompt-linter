import cohere

co = cohere.Client('yiOWD4KfXSiayGiim2MRmZRUvGsbdEFOY5QaCQ1Z')

def detox(input):
    response = co.generate(
        prompt="Rewrite this sentence positively: " + input,
        model="command-xlarge-beta",
        temperature=0.9
    )
    return response.generations[0].text
