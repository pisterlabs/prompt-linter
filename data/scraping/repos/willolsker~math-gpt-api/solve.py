import openai
import json
import functions
import dotenv
import os
dotenv.load_dotenv()


def run_conversation(word_problem):
    steps = []

    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system", "content": "Solve the problem provided by the user using a formula."},
            {"role": "user", "content": word_problem},
        ],
        functions=functions.functions,
        function_call="auto",
        temperature=0.2,
    )

    message = response["choices"][0]["message"]
    print(message["function_call"]["arguments"])

    if message.get("function_call"):
        function_name = message["function_call"]["name"]
        function_arguments = json.loads(message["function_call"]["arguments"])

        if function_name == "use_formula":
            display_values = [function_arguments["formula"]]
            for value in function_arguments["values"]:
                display_values.append(value["variable"] + " = " + value["value"])
            steps.append({
                "text": "Use the formula " + function_arguments["friendly_name"] + ".",
                "math": display_values
            })

            result = functions.substitute_values(function_arguments["formula"], function_arguments["values"])
            steps.append({
                "text": "Substitute the values into the formula.",
                "math": [result]
            })

            result = functions.use_formula(function_arguments["formula"], function_arguments["values"])
            steps.append({
                "text": "Simplify.",
                "math": [result]
            })

    return steps