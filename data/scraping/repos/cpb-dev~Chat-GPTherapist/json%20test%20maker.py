import openai, json

openai.api_key = 'Your API key'

prompt = """Your role is to create a JSON file for testing a program 
    that helps aid in finding out mental diagnosis based on a string of text from a client
    """

messages = [ { "role": "system", "content": prompt} ]

#params = input("What are the parametors you want to test? \n")
message = f"""Give 20 different text inputs that are written as if a therapist
    is talking about a patient and potential conditions they may have.
    Write this in extensive detail and more specific to actions the patient has
    exhibited to warrant such analysis. Use at least 100 words.
    """

messages.append(
    {"role": "user", "content": message}
)
chat = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", messages = messages
    )

reply = chat.choices[0].message.content
print(f"Done \n\n {reply}")

with open("sample.json", "w") as outfile:
    json.dump(reply, outfile)

test = input("OK?")
