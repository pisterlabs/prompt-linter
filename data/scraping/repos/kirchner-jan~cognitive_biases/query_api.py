import openai

openai.api_key = "API_KEY" # enter API KEY here
openai.api_base = "https://api.goose.ai/v1"

def get_completion(prompt , max_tokens=50 , stream_flag = False):
    return openai.Completion.create(engine="gpt-neo-20b", prompt=prompt, max_tokens=max_tokens , stream = stream_flag)
