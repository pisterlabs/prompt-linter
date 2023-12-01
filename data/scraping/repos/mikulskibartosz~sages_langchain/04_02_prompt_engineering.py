# Prompt engineering
## chain-of-thoughts

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

    first_template = """Review: Great place for pancakes and hot drinks. We received very good service from the staff and we liked the ambiance. I ordered the Salmon with pasta (tagliatelle) and was delighted.
Review Analysis: The review provides a positive rating of the restaurant, with the customer commending the food, service, and ambience. The reviewer specifically mentions ordering the salmon with pasta, indicating that they were satisfied with the dish.
Classification: Positive

Review: {review}
Review analysis: """

    prompt_template = PromptTemplate(
    input_variables=["review"],
    template=first_template
)

    review = "Perhaps it was tasty and good service in the past, but since they got so popular, they don’t care at all about the service."

    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    response = llm_chain.run(
        review=review,
        max_tokens=50,
        temperature=0.0,
    )

    print(response)

