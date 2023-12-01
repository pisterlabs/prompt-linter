""" create embeddings """
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
import openai
import tiktoken
import spacy
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)

# ceny gpt w dolarach

# MODEL = 'gpt-4'
# INPUT_PRICE = 0.03
# OUTPUT_PRICE = 0.06
# OUTPUT_TOKENS = 1500
# PART_MAX_TOKENS = 5000

# MODEL = 'gpt-3.5-turbo-16k'
# INPUT_PRICE = 0.003
# OUTPUT_PRICE = 0.004
# OUTPUT_TOKENS = 2000
# PART_MAX_TOKENS = 6000

MODEL = 'gpt-3.5-turbo'
INPUT_PRICE = 0.0015
OUTPUT_PRICE = 0.002
OUTPUT_TOKENS = 1000
PART_MAX_TOKENS = 2500

# api key
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY


@retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(6))
def get_answer_with_backoff(**kwargs):
    """ add exponential backoff to requests using the tenacity library """
    return openai.ChatCompletion.create(**kwargs)


def get_answer(prompt:str='', text:str='', model:str='gpt-4') -> str:
    """ funkcja konstruuje prompt do modelu GPT dostępnego przez API i zwraca wynik """
    result = ''
    prompt_tokens = completion_tokens = 0

    try:
        response = get_answer_with_backoff(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Jesteś pomocnym asystentem, specjalistą w dziedzinie historii, genealogii, życiorysów znanych postaci."},
                        {"role": "user", "content": f"{prompt}\n{text}"}
                    ],
                    temperature=0.0,
                    top_p = 1.0,
                    max_tokens=OUTPUT_TOKENS)

        result = response['choices'][0]['message']['content']
        prompt_tokens = response['usage']['prompt_tokens']
        completion_tokens = response['usage']['completion_tokens']

    except Exception as request_error:
        print(request_error)
        sys.exit(1)

    return result, prompt_tokens, completion_tokens


def count_tokens(text:str, model:str = "gpt-4") -> int:
    """ funkcja zlicza tokeny """
    num_of_tokens = 0
    enc = tiktoken.encoding_for_model(model)
    num_of_tokens = len(enc.encode(text))

    return num_of_tokens


# -------------------------------- MAIN ----------------------------------------
if __name__ == '__main__':

    # pomiar czasu wykonania
    start_time = time.time()

    # dane z pliku tekstowego
    print('Wczytanie pliku...')
    data_file = Path("..") / "data_psb_250" / 'Stanislaw_August_Poniatowski.txt'
    with open(data_file, "r", encoding='utf-8') as f:
        biogram_text = f.read()

    # spacy do podziału tekstu na zdania
    print('Podział na zdania...')
    nlp = spacy.load('pl_core_news_lg')

    doc = nlp(biogram_text)
    sentences = [sent.text for sent in doc.sents]
    first_sentence = sentences[0]

    print('Przygotowanie podziału tekstu na części mieszczące się w kontekście modelu...')
    parts = []
    part_tokens = 0
    part_num = 0
    for item in sentences:
        sent_tokens = count_tokens(item)
        if part_tokens + sent_tokens > PART_MAX_TOKENS:
            print(parts[part_num][:50] + '...')
            print(f'Tokenów w części: {part_tokens}')
            part_num += 1
            parts.append(item)
            part_tokens = sent_tokens
        else:
            if len(parts) == part_num + 1:
                parts[part_num] += ' ' + item
            else:
                parts.append(item)
            part_tokens += sent_tokens

    print(parts[part_num][:50] + '...')
    print(f'Tokenów w części: {part_tokens}')

    print('Przygotowanie streszczenia...')
    summaries = []
    total_tokens_in_prompt = 0
    total_tokens_in_completions = 0

    for part in parts:
        print(f'Streszczenie części: {part[:50] + "..."}')

        prompt = """ Na podstawie podanego tekstu będącego fragmentem biogramu
        Stanisława Augusta Poniatowskiego przygotuj streszczenie uwzględniając
        przede wszystkim informacje o krewnych lub powinowatych głównego
        bohatera. Możliwe pokrewieństwa/powinowactwa: ojciec, matka,
        syn, córka, brat, siostra, żona, mąż, teść, teściowa, dziadek, babcia, wnuk, wnuczka,
        szwagier, szwagierka, siostrzeniec, siostrzenica, bratanek, bratanica, kuzyn, kuzynka,
        zięć, synowa. Jeżeli to możliwe podawaj zawsze pełne imiona i nazwiska tych osób.
        Inne informacje nie są istotne. W tekście główny bohater może występować w formie
        skrótu od swojego nazwiska np.: S.
        Jeżeli w tekście nie ma danych o krewnych zwróć tylko i wyłącznie informację: brak danych.

        Tekst:
    """

        summary, tokens_in_prompt, tokens_in_completions = get_answer(prompt=prompt,
                                                                      text=part,
                                                                      model=MODEL)
        if not 'brak danych' in summary.strip().lower():
            summaries.append(summary)
        total_tokens_in_prompt += tokens_in_prompt
        total_tokens_in_completions += tokens_in_completions

    # skrócony biogram
    text = ' '.join(summaries)

    tokens_in_text = count_tokens(text)
    print(f'Tokenów w skróconym biogramie: {tokens_in_text}')

    output_file = Path("..") / "short_data_psb_250" / f'skrocony_biogram_relacje_rodzinne_by_{MODEL}_Stanislaw_August_Poniatowski.txt'
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write(text)

    # obliczenie kosztów
    price_gpt = (((total_tokens_in_prompt/1000) * INPUT_PRICE) +
                    ((total_tokens_in_completions/1000) * OUTPUT_PRICE))
    print(f'\nBiogram: {data_file}, koszt streszczenia: {price_gpt:.2f}')

    # czas wykonania programu
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Czas wykonania programu: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))} s.')
