import os 
import openai

os.environ['API_KEY'] = '<your api key goes here>'
openai.api_key = os.environ.get('API_KEY')

cot_messages = [
    {
        "role" : "system",
        "content" : (
                        "You are an interactive and friendly assistant for a bake-shop called 'COT' which is based in Lahore, Pakistan.\n"
                        "You have detailed knowledge about the various bakery items prepared at the bake-shop, their categories,\n"
                        "their ingredients, their nutritional information, and their prices.\n"
                    )
    },
    {
        "role" : "user",
        "content" : "List down all the bakery-items available at COT."
    }
]

cot_messages_formatted = "".join([f"{msg['role']} : {msg['content']}" for msg in cot_messages])

try:
    response = openai.completions.create(
        model = "text-davinci-003",
        prompt = cot_messages_formatted,
        max_tokens = 5000,
    )
    print(response.choices[0].text.strip())

except Exception as e:
    print(f"An ERROR recieved from the OpenAI's API: {e}")
