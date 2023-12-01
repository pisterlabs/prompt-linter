import openai, json

def function_a(number):
    print("Function A")
    print(number)

def function_b(text):
    print("Function B")
    print(text)

def function_c():
    print("Function C")

def run_conversation():
    # Step 1: send the conversation and available functions to GPT
    messages = [{"role": "user", "content": "Please call function B with a random quote from Hamlet. Paraphrase the response and return that paraphrased response."}]
    functions = [
        {
            "name": "function_a",
            "description": "Function A.",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "description": "An integer parameter that does nothing."
                    }
                }
            }
        },
        {
            "name": "function_b",
            "description": "Function B.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "A text input string."
                    }
                }
            }
        },
        {
            "name": "function_c",
            "description": "Function C.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    response_message = response["choices"][0]["message"]

    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):

        function_name = response_message["function_call"]["name"]
        function_args = json.loads(response_message["function_call"]["arguments"])

        function_response = None
        if function_name == "function_a":
            function_response = function_a(function_args.get("number"))
        elif function_name == "function_b":
            function_response = function_b(function_args.get("text"))
        elif function_name == "function_c":
            function_response = function_c()

        # Step 4: send the info on the function call and function response to GPT
        messages.append(response_message)  # extend conversation with assistant's reply
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )  # get a new response from GPT where it can see the function response

        return second_response

print(run_conversation())