import tiktoken
import openai
from ._algo import cos_summarize
from ._gpt_cache import judgment_cache

import os


PROMPT = ('You will be provided with an article from the website {url}. '
          'It has been cleaned to remove stopwords. '
          'Based on your knowledge of the website and this article, '
          'are them related to {keywords}? Output only yes or no. '
          '(yes if they relate to ANY of the keywords, and no otherwise)')

GPT3_TOKEN_LMT = 4050  # Max number of tokens GPT 3.5 can handle
GPT3_ENCODING = tiktoken.encoding_for_model('gpt-3.5-turbo')

openai.api_key = os.getenv('OPENAI_API_KEY')


@judgment_cache
def get_gpt_judgment(url: str, text: str, keywords: list[str]) -> str:
    """Return ChatGPT's judgment on whether the keywords relate to text & url
    """
    def get_token_n(text: str) -> int:
        return len(GPT3_ENCODING.encode(text))
        
    # Format the keywords to be GPT-readable
    # e.g., ['How', 'are', 'you'] -> "How", "are", or "you"
    if len(keywords) == 1:
        keywords = f'"{keywords[0]}"'
    else:
        rest = '", "'.join(keywords[:-1])
        keywords = f'"{rest}", or "{keywords[-1]}"'
    
    req_prompt = PROMPT.format(url=url, keywords=keywords)
    max_text_token_n = GPT3_TOKEN_LMT - get_token_n(req_prompt)
    
    if get_token_n(text) > max_text_token_n:
        # Summarize the text and extract the most relevant sentences
        summary = cos_summarize(text)
        summary = sorted(summary, key=summary.get)
        text = ''
        while summary:
            next_sent = summary.pop()
            if get_token_n(text + next_sent) >= max_text_token_n:
                break
            text += ' ' + next_sent

    try:
        return openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[{'role': 'system', 'content': req_prompt},
                      {'role': 'user', 'content': text}],
            temperature=0,
            max_tokens=2
        )['choices'][0]['message']['content'].lower()
    except openai.error.RateLimitError:
        # Max number of monthly requests reached
        return 'lim'
