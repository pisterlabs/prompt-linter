import tiktoken

#get_number_tokens_from_openai takes an input message and an encoding, and returns the number of tokens used
#this uses tiktoken, from openai
def get_number_tokens_from_openai(message: str, encoding: str) -> int:
    tokens = tiktoken.get_encoding(encoding).encode(message)
    return len(tokens)

#takes number of tokens used and the model used, returns the number of tokens left to be used, which can be used in the response

def how_many_tokens_remaining_as_int(tokens_used: int, model: str) -> int:
    token_limits = {
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-3.5": 4096,
        "gpt-3.5 (code-davinci-002)": 8001,
        "gpt-3.5 (turbo-16k)": 16384,
        "gpt-3": 2049,
    }

    return token_limits.get(model, -1) - tokens_used

# chooses an encode based on the chosen chat model
# there are different versions of gpt 3.5 that use different encoders - from tiktoken in model.py
def set_encoder(model: str) -> str:
    model_options = {
        "gpt-4": "cl100k_base",
        "gpt-4-32k": "cl100k_base",
        "gpt-3.5": "cl100k_base",
        "gpt-3.5 (code-davinci-002)": "p50k_base",
        "gpt-3.5 (turbo-16k)": "cl100k_base",
        "gpt-3": "r50k_base",
    }
    return model_options.get(model, "cl100k_base")
