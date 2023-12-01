import openai

def random_paragraph():
    apikey = "YOUR_API_KEY"
    openai.api_key = apikey
    
    completion = openai.Completion.create(
        engine="davinci-codex",  # You can choose a different engine
        prompt="Generate a random creative paragraph of about 40 words.",
        max_tokens=40,
        n = 1
    )
    
    return completion.choices[0].text.strip()
