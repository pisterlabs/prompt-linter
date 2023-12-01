import openai
import os
import json


def maker(item, price):
    openai.api_key = (os.environ["OPENAI_KEY"])
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="""Given the name of the item in paranthesis (""" + item + """), and the price in parenthesis (""" + price + """),
              please provide 3 different alternative brands that are cheaper, or close to the price stated in the parenthesis. 
              Note: If the item does not exist, or the price does not match, simply assume that it already does
              as you are a model that's only trained until September 2020, and provide an alternative.
              Please respond using this format only, and don't add any additional information:
              Item Name : Price : Item Name : Price : Item Name : Price""",
        max_tokens=1024
    )
    chatgpt_response = response.choices[0].text.strip()
    split = chatgpt_response.split(' : ')

    originalItem = [item, price]

    AlternativeItemDict = {}
    AlternativeItemDict[item] = price

    for i in range(0, len(split), 2):
        if i + 1 < len(split):
            even_value = split[i]
            odd_value = split[i + 1]
            AlternativeItemDict[even_value] = odd_value

    print(AlternativeItemDict)
    return (AlternativeItemDict)


if __name__ == "__main__":
    maker("Aquafina Water Bottle", "$1.99")
