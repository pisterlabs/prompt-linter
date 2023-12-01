import openai, json

client = openai.OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=json.load(open("key.json"))["openAiKey"],
)
#openai.api_key = json.load(open("key.json"))["openAiKey"]

#print(openai.api_key)

def openai_api(prompt):
    # Use the openai API to generate a response
    response = client.chat.completions.create(
        messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="gpt-3.5-turbo",)
    # Return the generated response
    return response.choices[0].message.content

print(openai_api("What is the smallest planet in the solar system?"))