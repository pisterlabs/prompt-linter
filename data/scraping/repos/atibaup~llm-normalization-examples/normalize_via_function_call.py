import os

import openai
from pydantic import BaseModel, Field

openai.api_key = os.getenv("OPENAI_API_KEY")


class NormalizedData(BaseModel):
    name: str = Field(..., description="The name of the person")
    age: int = Field(..., description="The age of the person")
    address: str = Field(..., description="The address of the person")


function_specs =  {
    "name": "normalize",
    "description": "Normalizes user input to the corresponding location and units",
    "parameters": NormalizedData.model_json_schema(by_alias=False)
}


def normalize(input: str) -> NormalizedData:
    normalized_data_fields = NormalizedData.model_fields.keys()
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
              "role": "system",
              "content": f"""
                You are a helpful assistant. Given a short biography of a person,
                you generate a JSON object with the following fields: {",".join(normalized_data_fields)}. 
              """
            },
            {
              "role": "user",
              "content": input
            }
        ],
        functions=[function_specs],
        function_call={"name": function_specs['name']},
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    function_args = response.choices[0]['message']['function_call']['arguments']
    normalized_parameters = NormalizedData.model_validate_json(function_args)
    return normalized_parameters


if __name__ == '__main__':
    #
    # Example usage
    # > python normalize_via_function_call.py "I was born 25 years ago in Kingston Ave, Brooklyn, NY, where I still live. My name is Patrick"
    import sys
    normalized = normalize(sys.argv[0])
    print(normalized)
