import openai

# Set up your OpenAI API key
api_key = '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'

# Initialize the OpenAI API client
openai.api_key = api_key

def generate_text(prompt, max_tokens=50, temperature=0.6):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].text.strip()

if __name__ == '__main__':
    prompt = "Once upon a time"
    generated_text = generate_text(prompt)

    print(generated_text)
