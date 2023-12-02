""" openai test - extraction info about parents, children, wife,
    husband from bio
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import openai
import spacy
import tiktoken


def get_data_gpt3(text:str='', query_prompt:str='') -> str:
    """ zwraca wynik zapytania do GPT-3 """
    result = ''

    response = openai.Completion.create(
         model="text-davinci-003",
         prompt=f"{query_prompt}\n\n {text}",
         temperature=0.0,
         max_tokens=900,
         top_p=1.0,
         frequency_penalty=0.8,
         presence_penalty=0.0)

    result = response['choices'][0]['text']

    return result


def short_version(text:str) -> str:
    """ short_version"""

    select_data = []
    words = ['ojciec', 'matka', 'syn', 'córka', 'brat', 'siostra', 'żona',
                'mąż', 'teść', 'teściowa', 'dziadek', 'babcia', 'wnuk', 'wnuczka',
                'szwagier', 'szwagierka', 'siostrzeniec', 'siostrzenica', 'bratanek',
                'bratanica', 'kuzyn', 'kuzynka', 'zięć', 'synowa', 'dziecko', 'wuj',
                'ciotka', 'rodzina', 'krewni', 'krewny', "ożenić", "bezdzietny", "ożeniony", "zamężna",
                "rodzic", "rodzice", "spokrewniony", "spokrewnieni", "małżeństwo", "rodzeństwo",
                "bratankowie", "siostrzeńcy", "bratanice", "siostrzenice", "małżeństwa"]

    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    # pierwszych pięć zdań
    select_data = sentences[0:5]

    # ze środkowych zdań tylko takie wskazujące na opisy relacji rodzinnych
    for i in range(5,len(sentences) - 5):
        sent_doc = nlp(sentences[i])
        for token in sent_doc:
            if token.lemma_ in words:
                select_data.append(sentences[i])
                break

    # ostatnie pięć zdań
    select_data += sentences[len(sentences) - 5:]

    result = ' '.join(select_data)
    return result


def count_tokens(text:str, model:str = "gpt2") -> int:
    """ funkcja zlicza tokeny """
    num_of_tokens = 0
    enc = tiktoken.get_encoding(model)
    num_of_tokens = len(enc.encode(text))

    return num_of_tokens


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    env_path = Path(".") / ".env"
    load_dotenv(dotenv_path=env_path)

    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    openai.api_key = OPENAI_API_KEY

    # spacy do podziału tekstu na zdania
    nlp = spacy.load('pl_core_news_md')

    # dane z pliku tekstowego
    data_folder = Path("..") / "data" / "psb_probki_200_txt_gpt3"

    data_file_list = data_folder.glob('*.txt')
    max_tokens = 3200

    licznik = 0
    for data_file in data_file_list:
        # ograniczona liczba biogramów
        licznik += 1
        if licznik > 50:
            break

        print(data_file)
        with open(data_file, 'r', encoding='utf-8') as f:
            data = f.read()

        data_file_name = os.path.basename(data_file)

        # jeżeli biogram jest zbyt duży dla modelu gpt-3 (4000 tokenów - 800
        # na odpowiedź) to biogram jest skracany do treści o największym prawdopodobieństwie
        # zawierania informacji o relacjach rodzinnych bohatera biogramu
        tokens_in_data = count_tokens(data, "gpt2")
        if len(data) > max_tokens:
            data = short_version(data)

        prompt = "Na podstawie podanego tekstu wyszukaj " \
                "wszystkich krewnych lub powinowatych głównego bohatera tekstu. " \
                "Możliwe rodzaje pokrewieństwa: ojciec, matka, syn, córka, brat, siostra, żona, mąż, teść, teściowa, dziadek, babcia, wnuk, wnuczka," \
                "szwagier, szwagierka, siostrzeniec, siostrzenica, bratanek, bratanica, kuzyn, kuzynka, zięć, synowa, teść bratanicy." \
                "Wynik wypisz w formie listy nienumerowanej, " \
                "w formie: główny bohater -> rodzaj pokrewieństwa -> osoba " \
                "Każda pozycja w osobnej linii. Na przykład: " \
                "- główny bohater -> brat -> Jan Kowalski" \
                "- główny bohater -> siostra -> Anna" \
                "Pomiń rodzaj pokrewieństwa jeżeli nie występuje w tekście. " \
                "Jeżeli w tekście nie ma żadnych informacji o pokrewieństwach głównego bohatera napisz: brak danych."

        file_output = Path("..") / "output" / "psb_probki_200_txt_gpt3" / data_file_name.replace('.txt', '.dane')
        with open(file_output, 'w', encoding='utf-8') as f:
            f.write(data)

        output = get_data_gpt3(data, prompt)

        file_output = Path("..") / "output" / "psb_probki_200_txt_gpt3" / data_file_name.replace('.txt', '.relacje')
        with open(file_output, 'w', encoding='utf-8') as f:
            f.write(output + '\n')
