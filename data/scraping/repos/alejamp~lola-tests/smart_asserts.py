import os
import openai
from thefuzz import fuzz
from config import config

# load key from env var
openai.api_key= config["OPENAI_API_KEY"]



def nlp_assert(text_to_analyze, question, expected, treshold=90):
    prompt = question + "\n" + text_to_analyze + "\n"
    messages = [
        # content= question + "\n" + response + "\n",
        {"role": "system", "content": f"{text_to_analyze}"},
        {"role": "user", "content": f"{question}"},
    ]
    res = openai.ChatCompletion.create(
        model=config["OPENAI_MODEL"],
        messages=messages,
        temperature=0,
        top_p=0,
        # max_tokens=100
    )
    output = res["choices"][0]["message"]["content"] # type: ignore
    # fuzzy compare case insensitive
    match = fuzz.ratio(output.lower(), expected.lower())
    return (match > treshold, output, expected)





if __name__ == "__main__":
    inputText = "The capital of France is Paris."
    question = "What is the capital of France? Give only the city name."
    expected = "Paris"
    result, o, e = nlp_assert(inputText, question, expected)
    print("Input text:", inputText)
    print("Question:", question)
    print("Expected:", expected)
    print("Output:", o)
    print("Match:", result)
    print("---------------------------------------------------------")



    inputText = "I'm sorry but Argentinian IDs are not allowed"
    question = "ID was rejected? Answer only true or false"
    expected = "true"
    result, o, e = nlp_assert(inputText, question, expected)
    print("Input text:", inputText)
    print("Question:", question)
    print("Expected:", expected)
    print("Output:", o)
    print("Match:", result)
    print("---------------------------------------------------------")

