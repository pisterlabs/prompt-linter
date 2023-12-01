import os

import openai
from dotenv import load_dotenv

OUTPUT_RECIPE_FUNCTION = {
    "name": "output_recipe",
    "description": "レシピを出力する",
    "parameters": {
        "type": "object",
        "properties": {
            "ingredients": {
                "type": "array",
                "description": "材料",
                "items": {"type": "string"},
                "examples": [["鶏もも肉 300g", "玉ねぎ 1個"]],
            },
            "steps": {
                "type": "array",
                "description": "手順",
                "items": {"type": "string"},
                "examples": [["材料を切ります。", "材料を炒めます。"]],
            },
        },
        "required": ["ingredients", "steps"],
    },
}


def main():
    load_dotenv()

    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "カレーのレシピを教えてください。"}],
        functions=[OUTPUT_RECIPE_FUNCTION],
        function_call={"name": OUTPUT_RECIPE_FUNCTION["name"]},
    )

    response_message = response["choices"][0]["message"]
    function_call_name = response_message["function_call"]["name"]
    function_call_args = response_message["function_call"]["arguments"]

    print(function_call_name)
    print(function_call_args)


if __name__ == "__main__":
    main()
