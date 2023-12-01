import openai
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Initialize the OpenAI API client
openai.api_key = "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"


def generate_summary(keywords, topic_description):
    chat = ChatOpenAI(openai_api_key = "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")
    prompt = f"Please generate a summary based on the provided keywords: {', '.join(keywords)}. The topic revolves around {topic_description}."
    result = chat(
        [
            HumanMessage(
                content=prompt
            )
        ]
    )
    # Process the AI response or handle the result here
    # return "this works too!"
    return result


def generate_example(keywords, topic_description):
    # Implement example generation using GPT-3.5
    # Use keywords and topic_description to generate an example
    #  API call:
    prompt = f"Provide an example related to: Keywords: {', '.join(keywords)}. Topic description: {topic_description}"
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=100
    )
    example = response.choices[0].text.strip()
    # return "examples work too!"
    return example

