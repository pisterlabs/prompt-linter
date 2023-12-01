import os

import openai
from mastodon import Mastodon, AttribAccessDict
from typing import Iterable

def mastodon_client(credential_file : str, token_file : str) -> Mastodon:
    # Create a new app if necessary
    if not os.path.exists(credential_file):
        Mastodon.create_app(
            'sentiment_nlp',
            api_base_url = os.environ['MASTODON_BASE_URL'],
            to_file = credential_file
        )

    # Log in
    mastodon = Mastodon(
        client_id = credential_file,
        api_base_url = os.environ['MASTODON_BASE_URL']
    )

    mastodon.log_in(
        os.environ['MASTODON_USER'],
        os.environ['MASTODON_PASS'],
        to_file = token_file
    )

    return mastodon

def extract_contents(tl : Iterable[AttribAccessDict]) -> list[str]:
    contents = []
    for toot in tl:
        # `toot` is a `mastodon.Mastodon.AttribAccessDict`
        # `content` is HTML formatted, so ultimately you might want to strip tags, but I think OpenAI can handle it
        contents.append(toot['content'])
    return contents

def sentiment_nlp(content : str) -> str:
    openai.api_key = os.environ['OPENAI_KEY']
    # OpenAI prompt for sentiment analysis
    prompt = f"""Label the sentiment of this sentence:\n\n{content}\n\nPositive\nNeutral\nNegative\n\nLabel:"""
    response = openai.Completion.create(
        engine = 'text-davinci-002',
        prompt = prompt,
        temperature = 0,
        max_tokens = 1,
        top_p = 1.0,
        frequency_penalty = 0.0,
        presence_penalty = 0.0,
        best_of = 1
    )
    return response['choices'][0]['text'].strip()

def main():
    credential_file = 'sentiment_nlp_clientcred.secret'
    token_file = 'sentiment_nlp_usercred.secret'

    mastodon = mastodon_client(credential_file, token_file)

    tl = mastodon.timeline('local', limit=10)
    contents = extract_contents(tl)
    sentiments = []
    for content in contents:
        sentiment = sentiment_nlp(content)
        sentiments.append(sentiment)

    for (content,sentiment) in zip(contents,sentiments):
        print(f"{sentiment} : {content}")


if __name__ == '__main__':
    main()