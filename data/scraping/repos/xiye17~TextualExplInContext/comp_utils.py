import os
import openai
from transformers import GPT2TokenizerFast

_TOKENIZER = GPT2TokenizerFast.from_pretrained('gpt2')
GPT3_LENGTH_LIMIT = 2049
openai.api_key = os.getenv("OPENAI_API_KEY")

def gpt_style_tokenize(x):
    return _TOKENIZER.tokenize(x)

def length_of_prompt(prompt, max_tokens):
    return len(_TOKENIZER.tokenize(prompt)) + max_tokens

def safe_completion(engine, prompt, max_tokens, stop, temp=0.0, logprobs=5):
    len_prompt_token = len(_TOKENIZER.tokenize(prompt))    
    if max_tokens + len_prompt_token >= GPT3_LENGTH_LIMIT:
        print("OVERFLOW", max_tokens + len_prompt_token)
        return {
            "text": "overflow"
        }
    resp = openai.Completion.create(engine=engine, prompt=prompt, max_tokens=max_tokens, stop=stop,
        temperature=0.0, logprobs=logprobs, echo=True)

    pred = resp["choices"][0]
    return pred        

def conditional_strip_prompt_prefix(x, p):
    if x.startswith(p):
        x = x[len(p):]
    return x.strip()
