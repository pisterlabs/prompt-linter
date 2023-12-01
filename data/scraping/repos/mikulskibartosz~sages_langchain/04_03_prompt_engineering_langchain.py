# Prompt engineering
## chain-of-thoughts

from langchain.llms import OpenAI
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain import LLMChain


if __name__ == '__main__':
    with open('api.key', 'r') as openai_api_key:
        openai_api_key = openai_api_key.read().strip()

    llm = OpenAI(
        model_name="text-davinci-003",
        openai_api_key=openai_api_key
    )

    examples = [
        {
            "review": "Great place for pancakes and hot drinks. We received very good service from the staff and we liked the ambiance. I ordered the Salmon with pasta (tagliatelle) and was delighted.",
            "analysis": "The review provides a positive rating of the restaurant, with the customer commending the food, service, and ambience. The reviewer specifically mentions ordering the salmon with pasta, indicating that they were satisfied with the dish.",
            "classification": "positive"
        }
    ]

    examples_template = """Review: {review}
Review Analysis: {analysis}
Classification: {classification}

Review: {review}
Review analysis: """

    example_prompt = PromptTemplate(
        template=examples_template,
        input_variables=["review", "analysis", "classification"]
    )

    prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    input_variables=["review"],
    suffix="""Review: {review}
Review Analysis: """,
)

    review = "Perhaps it was tasty and good service in the past, but since they got so popular, they don’t care at all about the service."

    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    response = llm_chain.run(
        review=review,
        max_tokens=50,
        temperature=0.0,
    )

    print(response)

