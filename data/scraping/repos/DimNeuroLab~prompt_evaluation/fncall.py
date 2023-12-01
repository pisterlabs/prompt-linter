import json
from utils import get_api_key
import openai

CLASSIFICATION_PROMPT = """
Given the following feature:
    {feature_description}

The feature is present in the following prompt:
    {positive_few_shot}

The feature is not present in the following prompt:
    {negative_few_shot}

Tell me whether the feature is present in the prompt given below. Formalize your output as a json object, 
where the key is the feature description and the associated value is 1 if the feature is present or 0 if not.

Prompt:
    {eval_prompt}
"""

CLASSIFICATION_FUNCTION = [
    {
        "name": "feature_present",
        "description": "Is a particular feature present in a piece of text given positive and negative examples",
        "parameters": {
            "type": "object",
            "properties": {
                "classification": {
                    "type": "boolean",
                    "description": "A classification of whether the feature is present",
                }
            },
            "required": ["classification"],
        },
    }
]


def classify_feature_presence(
        feature_description: str,
        positive_few_shot: str,
        negative_few_shot: str,
        eval_prompt: str,
        model: str
) -> bool:
    messages = [
        {
            "role": "user",
            "content": CLASSIFICATION_PROMPT.format(
                feature_description=feature_description,
                positive_few_shot=positive_few_shot,
                negative_few_shot=negative_few_shot,
                eval_prompt=eval_prompt
            )
        }
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        functions=CLASSIFICATION_FUNCTION,
    )

    json_response = response["choices"][0]["message"]["function_call"]["arguments"]
    object_response = json.loads(json_response)

    return object_response["classification"]


def _test():
    FEATURE_DESCRIPTION = "instructing the language model to ask back questions"
    POSITIVE_FEW_SHOT = "address me as 'Dear' act as: teacher with a sense of humor explain one aspect of how social media\
        algorithms work ask me one short question to assess my learning wait for my answer give feedback about my answer\
        later explain the next point about how social media algorithms work follow this loop until you explain all the aspects"
    NEGATIVE_FEW_SHOT = "I want you to teach me the disadvantages of social media according to my personal information like age, level of education, & culture."
    EVAL_PROMPT = "Please teach me the disadvantages of social media considering my age, level of education, and culture in an interactive manner.\
        Doing this, try to be more interactive by asking me questions."
    MODEL = "gpt-3.5-turbo"

    feature_is_present = classify_feature_presence(
        feature_description=FEATURE_DESCRIPTION,
        positive_few_shot=POSITIVE_FEW_SHOT,
        negative_few_shot=NEGATIVE_FEW_SHOT,
        eval_prompt=EVAL_PROMPT,
        model=MODEL
    )

    print(feature_is_present)

    EVAL_PROMPT = "Please teach me what are some disadvantages of using social media in an interactive manner by considering my age, educational level, & culture."

    feature_is_present = classify_feature_presence(
        feature_description=FEATURE_DESCRIPTION,
        positive_few_shot=POSITIVE_FEW_SHOT,
        negative_few_shot=NEGATIVE_FEW_SHOT,
        eval_prompt=EVAL_PROMPT,
        model=MODEL
    )

    print(feature_is_present)


if __name__ == "__main__":
    openai.api_key = get_api_key()

    _test()
