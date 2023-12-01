# !pip install python-dotenv    # https://pypi.org/project/python-dotenv/
from dotenv import dotenv_values
import openai   # https://pypi.org/project/openai/
# pip install openai
import json  # https://docs.python.org/3/library/json.html


config = dotenv_values(".env")
openai.api_key = config["OPENAI_KEY"]


def pizza_info(pizza_style, quantity):
    # print(f'pizza_style = {pizza_style}')
    # print(f'quantity = {quantity}')

    struct1 = {
        "name": pizza_style,
        "price": float(quantity) * 10.99,
    }
    if pizza_style.lower() == "mushroom":
        struct1["price"] = float(quantity) * 9.99

    return json.dumps(struct1)


def salad_info(salad_name, quantity):
    # print(f'salad_name = {salad_name}')
    # print(f'quantity = {quantity}')

    struct1 = {
        "name": salad_name,
        "price": float(quantity) * 4.95,
    }
    if salad_name.lower() == "caesar":
        struct1["price"] = float(quantity) * 6.49

    return json.dumps(struct1)


def pizza_salad_info(pizza_style, pizza_quantity, salad_name, salad_quantity):
    # print(f'pizza_style = {pizza_style}')
    # print(f'pizza_quantity = {pizza_quantity}')
    # print(f'salad_name = {salad_name}')
    # print(f'salad_quantity = {salad_quantity}')

    struct1 = {
        "pizza_name": pizza_style,
        "pizza_price": float(pizza_quantity) * 10.99,
        "salad_name": salad_name,
        "salad_price": float(salad_quantity) * 4.95,
    }
    if pizza_style.lower() == "mushroom":
        struct1["pizza_price"] = float(pizza_quantity) * 9.99
    if salad_name.lower() == "caesar":
        struct1["salad_price"] = float(salad_quantity) * 6.49

    return json.dumps(struct1)


def add_prices(price1, price2):
    # print(f'price1 = {price1}')
    # print(f'price2 = {price2}')

    struct1 = {
        "price": price1+price2,
    }

    return json.dumps(struct1)


def run_conversation(query):
    print(f'Q: {query}')
    message = get_one_response_with_functions(query)
    if message.get("function_call"):
        function_name = message["function_call"]["name"]
        arguments = json.loads(message["function_call"]["arguments"])
        amount = arguments.get("quantity")
        if function_name == "get_pizza_info":
            pizza_name = arguments.get("pizza_name")
            function_response = pizza_info(
                pizza_style=pizza_name,
                quantity=amount
            )
            # print(f'function_response = {function_response}')
        elif function_name == "get_salad_info":
            salad_name = arguments.get("salad_name")
            function_response = salad_info(
                salad_name=salad_name,
                quantity=amount
            )

        elif function_name == "pizza_plus_salad_info":
            pizza_name = arguments.get("pizza_name")
            pizza_number = arguments.get("pizza_number")
            salad_name = arguments.get("salad_name")
            salad_number = arguments.get("salad_number")
            function_response = pizza_salad_info(
                pizza_style=pizza_name,
                pizza_quantity=pizza_number,
                salad_name=salad_name,
                salad_quantity=salad_number
            )
            # print(f'function_response = {function_response}')
        else:
            price1 = arguments.get("price1")
            price2 = arguments.get("price2")
            function_response = add_prices(
                price1=price1,
                price2=price2
            )
            # print(f'function_response = {function_response}')
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {"role": "user", "content": query},
                message,
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
            ],
        )
        return f'Function: {function_name}: {second_response["choices"][0]["message"]["content"]}'

    return f'Direct: {message["content"]}'


def get_one_response_with_functions(query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": query}],
        functions=[
            {
                "name": "get_pizza_info",
                "description": "Get various info about a pizza, including price.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pizza_name": {
                            "type": "string",
                            "description": "The name of the pizza to get info about, e.g. \"pepperoni\".",
                        },
                        "quantity": {
                            "type": "number",
                            "description": "Number of pizzas to get price about, e.g. 3",
                        },
                    },
                    "required": ["pizza_name", "quantity"],
                }
            },
            {
                "name": "get_salad_info",
                "description": "Get various info about salads, including price.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "salad_name": {
                            "type": "string",
                            "description": "The name of the salad to get info about, e.g. \"caesar\".",
                        },
                        "quantity": {
                            "type": "number",
                            "description": "Number of salads to get price about, e.g. 3",
                        },
                    },
                    "required": ["salad_name", "quantity"],
                }
            },
            {
                "name": "add_prices",
                "description": "Combine the prices of two items by adding them up.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "price1": {
                            "type": "number",
                            "description": "The price of the first item.",
                        },
                        "price2": {
                            "type": "number",
                            "description": "The price of the second item.",
                        },
                    },
                    "required": ["price1", "price2"],
                }
            },
            {
                "name": "pizza_plus_salad_info",
                "description": "Combine the prices and quantities of two items by adding them up.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pizza_name": {
                            "type": "string",
                            "description": "The name or style of the pizza.",
                        },
                        "pizza_number": {
                            "type": "number",
                            "description": "The number of pizzas.",
                        },
                        "salad_name": {
                            "type": "string",
                            "description": "The name or style of the salad.",
                        },
                        "salad_number": {
                            "type": "number",
                            "description": "The number of salads.",
                        },
                    },
                    "required": ["pizza_name", "pizza_number", "salad_name", "salad_number"],
                }
            }
        ],
        function_call="auto"
    )

    message = response["choices"][0]["message"]
    return message


for prompt in [
    # "Be as brief as possible. What is the capital of Sweden? Of France?, Of Spain?",
    # "Be as brief as possible. Price for 3 salmon pizzas? For 2 caesar salads?",
    # "Be as brief as possible. Price for 2 caesar salads? For 3 salmon pizzas?",
    # "Be as brief as possible. What's the cost for 4 shrimp salads?",
    # "Be as brief as possible. Calculate the price for 3 salmon pizzas plus a mushroom salad?",
    # "Be as brief as possible. How to calculate the cost of two items? Be brief.",
    "Be as brief as possible. I ordered 3 pepperoni pizza and 3 shrimp salad. What is the total?",
    # "Be as brief as possible. The price for 3 salmon pizzas is $32.97. Now let's calculate the price for a mushroom salad. What's the total?",
    # "Be as brief as possible. How expensive is a pepperoni pizza?",
    # "Be as brief as possible. How much for two four cheese pizzas?",
    "Be as brief as possible, no explanation, just the final dollar amount. The cost of a mushroom pizza is $12.99. Calculate the cost of 2 pepperoni pizzas. What is the total for all mushroom and pepperoni the pizzas?"
]:
    output = run_conversation(prompt)
    print(f'A: {output}\n')
