# Alternatywne implementacje AI w Langchain,

from langchain import HuggingFacePipeline
from langchain import PromptTemplate
from langchain import LLMChain


if __name__ == '__main__':
    with open('api.key', 'r') as openai_api_key:
        openai_api_key = openai_api_key.read().strip()

    llm = HuggingFacePipeline.from_model_id(
        model_id="gpt2",
        task="text-generation",
        model_kwargs={"temperature":0.1},
        pipeline_kwargs={"do_sample":True, "max_new_tokens":50}
    )

    template = """Review: {review}

    Classify the review ({options}):"""

    prompt_template = PromptTemplate(
    input_variables=["review", "options"],
    template=template
)

    review = "Huge museum that has a ton of items stolen from colonized people and their lands. The information is very insightful and you can spend hours in the museum learning and still need to come again to see  more. The exhibits have plenty to offer and the best part is the museum is free. I agree that the items should be returned to their rightful owners!!"

    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    response = llm_chain.run(
        review=review,
        options="positive/negative",
        stop="\n"
    )

    print(response)
