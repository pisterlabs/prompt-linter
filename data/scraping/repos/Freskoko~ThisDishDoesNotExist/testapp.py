import openai

if True:
    openai.api_key = "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"

def generate_prompt(food):
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=(f"create a name for a crazy complex new type of {food}"),
        temperature=0.9,
        max_tokens = 200,
        presence_penalty = 2.0,
    )
    return (response.choices[0].text)

if __name__ == "__main__":
    print(generate_prompt("cake"))




