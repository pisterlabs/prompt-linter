import os 
import openai

os.environ['API_KEY'] = '<your api key goes here>'
openai.api_key = os.environ.get('API_KEY')

messages = [
    {
        "role" : "system",
        "content" : (
                        "You are a interactive and friendly assistant for a bake-shop called 'COT' which is based in Lahore, Pakistan.\n"
                        "You have detailed knowledge about the various bakery items prepared at the bake-shop, their categories,\n"
                        "their ingredients, their nutritional information, and their prices.\n"
                    )
    },
    {
        "role" : "user",
        "content" : "Please provide the detailed nutritional information of brownies."
    },
    {
        "role" : "assistant",
        "content" : "You start every conversation with the phrase: 'Hola! Welcome to COT :)\n'"
    },
]

formatted_prompt = "".join([f"{msg['role']} : {msg['content']}" for msg in messages])

response = openai.completions.create(
    model = "text-davinci-003",
    prompt = formatted_prompt,
    max_tokens = 500,
)

messages.append(
    {
        "role" : "assistant", 
        "content" : response.choices[0].text.strip()
    }
)

messages.append(
    {
        "role" : "user",
        "content" : "And what about cup-cakes?"
    }
)

formatted_prompt = "".join([f"{msg['role']} : {msg['content']}" for msg in messages])

response = openai.completions.create(
    model = "text-davinci-003",
    prompt = formatted_prompt,
    temperature = 0.1,
    # temperature = 1.9,
    max_tokens = 500,
)

print(response.choices[0].text.strip());
