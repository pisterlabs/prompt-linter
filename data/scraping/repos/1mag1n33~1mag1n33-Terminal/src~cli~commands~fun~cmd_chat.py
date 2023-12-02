import openai

openai.api_key = "no"

def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=3000,
        n=1,
        stop=None,
        temperature=0.5,
    )

    return response.choices[0].text.strip()

def do_chat(self, args):
    """Allows you to chat with chatgbt"""
    print("Enter your prompt below ('exit' to quit):")
    while True:
        prompt = input("> ")

        if prompt.lower() == "exit":
            break

        response = generate_response(prompt)
        print(response)