# Prompt engineering
## few-shot learning (in-context learning)

import json
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain import LLMChain


if __name__ == '__main__':
    with open('api.key', 'r') as openai_api_key:
        openai_api_key = openai_api_key.read().strip()

    llm = OpenAI(
        model_name="text-davinci-003",
        openai_api_key=openai_api_key
    )

    template = """Review: They have very kind and polite workers, i don’t really know about the prices because all i bought was some bottles of beer.
{{"classification": "positive"}}

Review: Perhaps it was tasty and good service in the past, but since they got so popular, they don’t care at all about the service.
{{"classification": "negative"}}

Review: {review}
"""

    prompt_template = PromptTemplate(
    input_variables=["review"],
    template=template
)

    review = "Great place for pancakes and hot drinks. We received very good service from the staff and we liked the ambiance. I ordered the Salmon with pasta (tagliatelle) and was delighted."

    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    response = llm_chain.run(
        review=review,
        max_tokens=50,
        temperature=0.0,
        stop="\n"
    )

    print(response)

    parsed_json = json.loads(response)
    print(parsed_json["classification"])