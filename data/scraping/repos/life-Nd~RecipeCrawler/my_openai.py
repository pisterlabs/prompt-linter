import openai
from api_keys import ApiKeys

openai.api_key = ApiKeys.openai_api_key
print("\nAI: Hey, what's the question? ")
search = input("Me: ")
list_engines = ["text-davinci-002", "text-curie-001", "text-babbage-001",
                "text-ada-001", "code-davinci-002", "code-cushman-001"]
engine = list_engines[0]
temp = 0.5
toxens = 256

if search:
    response = openai.Completion.create(
        engine=engine,
        prompt=search,
        temperature=temp,
        max_tokens=toxens,
    )


    text =str(response["choices"][0]["text"]).strip()
    print("AI :" +text)
