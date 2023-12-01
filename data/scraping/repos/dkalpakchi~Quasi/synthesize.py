import os
import json
import time

import dotenv
import openai
import requests
import jsonlines as jsl

from tqdm import tqdm


# Each tuple is 
#   (
#      total number of words, 
#      average number of words per sentence, 
#      average number of chars per word
#   )
# The categories in Skolverket's data (and their corresponding length stats are:
# => instruction: default
# => e-mail: default
# => info: wiki
# => post: default
# => ads: default
# => forum: forum
# => review: default
# => news: news


AVERAGE_LENGTHS = {
    'forum': {
        'familjeliv': (885 * 10**6, 12.56, 4.51),
        'flashback': (711 * 10**6, 12.92, 4.67)
    },
    'blogs': {
        'bloggmix': (375 * 10**6, 13.82, 4.69)
    },
    'news': {
        'webbnyheter': (87 * 10**6, 15.31, 5.42),
        'svt': (179 * 10**6, 13.65, 5.40)
    },
    'info': {
        'wiki': (314 * 10**6, 10.67, 5.63)
    }
}


def calc_default():
    triples = [y for x in AVERAGE_LENGTHS.values() for y in x.values()]
    N = sum([x[0] for x in triples])
    avg_words, avg_chars = 0, 0
    for n, w, c in triples:
        weight = n / N
        avg_words += weight * w
        avg_chars += weight * c
    return {'default': (N, avg_words, avg_chars)}


def calc_avg_sent_char_len(triple_dct):
    triples = triple_dct.values()
    N = sum([x[0] for x in triples])
    avg_words, avg_chars = 0, 0
    for n, w, c in triples:
        weight = n / N
        avg_words += weight * w
        avg_chars += weight * c
    return avg_words * avg_chars

AVERAGE_LENGTHS['default'] = calc_default()


if __name__ == '__main__':
    dotenv.load_dotenv()

    openai.api_key = os.getenv('SECRET_KEY')

    generated = []

    default_prompt = "Skriv {} olika läsförståelsefrågor med 4 alternativ (a, b, c, och d) och ge varje fråga en unik nummer (1, 2, 3, osv). Första alternativet (a) ska alltid vara rätt, medan de andra alternativen (b, c, och d) ska vara felaktiga, men troliga. Alla frågor måste kunna besvaras av den följande texten. Ordna frågor från den lättaste till den svåraste."

    records = jsl.Reader(open('skolverket.jsonl'))

    ts = int(time.time())

    records = list(records)

    with jsl.open('generated_{}.jsonl'.format(ts), 'w') as writer:
        for record in tqdm(records):
            text = record['text']
            triples_dct = AVERAGE_LENGTHS.get(
                record['category'],
                AVERAGE_LENGTHS['default']
            )
            avg_sent_chars = calc_avg_sent_char_len(triples_dct)
            num_q = int(len(text.strip()) / avg_sent_chars)
            
            while True:
                try:
                    gen_params = {
                        'prompt': '{}\n\n{}\n\n'.format(
                            default_prompt.format(num_q), text
                        ),
                        'temperature': 0.7,
                        'max_tokens': 2048
                    }
                    completion = openai.Completion.create(engine='text-davinci-003', **gen_params)
                except openai.error.RateLimitError:
                    time.sleep(60)
                    continue

                writer.write({
                    'text': text,
                    'requested_q': num_q,
                    'params': gen_params,
                    'res': completion
                })

                break
