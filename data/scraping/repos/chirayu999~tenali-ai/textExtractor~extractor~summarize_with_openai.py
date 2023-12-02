import openai


def summarize_with_openai(prompt, text):

    openai.api_key = 'YOUR_OPENAI_API_KEY'

    full_prompt = f'{prompt}\nText: "{text}"\n'

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=full_prompt,
        max_tokens=150
    )

    return response.choices[0].text.strip()
