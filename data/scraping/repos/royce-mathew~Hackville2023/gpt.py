import cohere
import config as config


co = cohere.Client(config.COHERE_API_KEY);

def evaluate_input(input_sentence: str):
    response = co.generate( 
        model='xlarge', 
        prompt = input_sentence,
        max_tokens=200, 
        temperature=0.8,
    )

    print(response.generations)

    if (response.generations[0]):
        return response.generations[0].text
    else:
        return "An Exception Occured"

