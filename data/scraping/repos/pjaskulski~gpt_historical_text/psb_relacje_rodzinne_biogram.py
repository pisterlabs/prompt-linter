""" openai test - extraction info about parents, children, wife,
    husband from bio
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import openai
import spacy
import tiktoken


def get_answer(model:str='gpt-4', text:str='', prompt:str='') -> str:
    """ funkcja konstruuje prompt do modelu GPT dostępnego przez API i zwraca wynik """
    result = ''

    response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Jesteś pomocnym asystentem, specjalistą w dziedzinie historii, genealogii, życiorysów znanych postaci."},
                    {"role": "user", "content": f"{prompt}\n\nTekst:\n\n{text}"}
                ],
                temperature=0.0,
                top_p = 1.0)

    result = response['choices'][0]['message']['content']

    return result


def short_version(text:str) -> str:
    """ short_version"""

    select_data = []
    words = ['ojciec', 'matka', 'syn', 'córka', 'brat', 'siostra', 'żona',
                'mąż', 'teść', 'teściowa', 'dziadek', 'babcia', 'wnuk', 'wnuczka',
                'szwagier', 'szwagierka', 'siostrzeniec', 'siostrzenica', 'bratanek',
                'bratanica', 'kuzyn', 'kuzynka', 'zięć', 'synowa', 'dziecko', 'wuj',
                'ciotka', 'rodzina', 'krewni', 'krewny', "ożenić", "bezdzietny", "ożeniony",
                "zamężna", "stryj", "stryjenka", "wujenka",
                "rodzic", "rodzice", "spokrewniony", "spokrewnieni", "małżeństwo", "rodzeństwo",
                "bratankowie", "siostrzeńcy", "bratanice", "siostrzenice", "małżeństwa, wyszła"]

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
    data_file_list = ['../data/psb_probki_200_txt_gpt3/Jadwiga_Jagiellonka.txt']

    licznik = 0
    for data_file in data_file_list:
        print(data_file)

        # wczytanie tekstu z podanego pliku
        text_from_file = ''
        with open(data_file, 'r', encoding='utf-8') as f:
            text_from_file = f.read().strip()

        if not text_from_file:
            print('Brak tekstu w pliku:', data_file)
            continue

        data_file_name = os.path.basename(data_file)

        # jeżeli biogram jest zbyt duży dla modelu gpt-4 (8000 tokenów - 1000
        # na odpowiedź) to biogram jest skracany do treści o największym prawdopodobieństwie
        # zawierania informacji o relacjach rodzinnych bohatera biogramu
        # jeżeli plik ze skróconymi danymi już istnieje to dane są wczytywane z dysku
        max_tokens = 7000

        tokens_in_data = count_tokens(text_from_file, "gpt2")
        print('Liczba tokenów:', tokens_in_data)
        if tokens_in_data > max_tokens:
            file_output = Path("..") / "output" / "gpt-4-api-dane" / data_file_name.replace('.txt', '.dane')
            if os.path.exists(file_output):
                with open(file_output, 'r', encoding='utf-8') as f:
                    text_from_file = f.read()
            else:
                text_from_file = short_version(text_from_file)
                with open(file_output, 'w', encoding='utf-8') as f:
                    f.write(text_from_file)

        # ///
        # Przykład 2: "Sapieha Jan Fryderyk h. Lis (1618–1664), pisarz polny kor. Był wnukiem woj.
        # witebskiego Mikołaja (zob.), najstarszym synem podkomorzego włodzimierskiego
        # Fryderyka (zm. 1626) i Ewy ze Skaszewskich, bratem oboźnego lit. Tomasza Kazimierza
        # (zob.), bpa wileńskiego Aleksandra Kazimierza i krajczego lit. Krzysztofa Franciszka."
        # Wynik:
        # [{"relacja":"dziadek", "osoba":"Mikołaj Sapieha"},
        # {"relacja":"ojciec", "osoba":"Fryderyk Spaieha"},
        # {"relacja":"matka", "osoba":"Ewa ze Skaszewskich"},
        # {"relacja":"brat", "osoba":"Tomasz Kazimierz Sapieha"},
        # {"relacja":"brat", "osoba":"Aleksander Kazimierz Sapieha"},
        # {"relacja":"brat", "osoba":"Krzysztof Franciszek Sapieha"}
        # ]

        prompt_template = """Na podstawie podanego tekstu wyszukaj wszystkich krewnych lub powinowatych głównego bohatera tekstu: {name}. Możliwe rodzaje pokrewieństwa: ojciec, matka, syn, córka, brat, siostra, żona, mąż, teść, teściowa, dziadek, babcia, wnuk, wnuczka, szwagier, szwagierka, siostrzeniec, siostrzenica, bratanek, bratanica, kuzyn, kuzynka, zięć, synowa.
Wynik przedstaw w formie listy obiektów JSON zawierających pola:
relacja: rodzaj pokrewieństwa (kim osoba była dla bohatera/bohaterki )
osoba: nazwa (imię i nazwisko osoby związanej relacją z bohaterem)
Wypisz tylko rodzaje pokrewieństwa, które występują w tekście.
Jeżeli w tekście nie ma żadnych informacji o pokrewieństwach głównego bohatera napisz: brak danych.

Przykład: "Soderini Carlo (ok. 1537–1581), kupiec i bankier. Był jednym z pięciu synów Niccola i Annaleny Ricasoli, młodszym bratem Bernarda (zob.).
Jego bratanicą była Małgorzata Anna, żona Winfrida de Loeve. S. ożenił się z Joanną, córką burgrabiego krakowskiego Adama Kurozwęckiego."
Wynik:
[{"relacja":"ojciec", "osoba":"Niccolo"},
 {"relacja":"matka": "osoba":"Annalena Ricasoli"},
 {"relacja":"brat": "osoba":"Bernard"},
 {"relacja":"bratanica": "osoba":"Małgorzata Anna"},
 {"relacja":"żona": "osoba":"Joanna"},
 {"relacja":"teść": "osoba":"Adam Kurozwęcki"}
]

Tekst:
"""

        output = get_answer(model="gpt-4",
                            text=text_from_file,
                            prompt=prompt_template)

        file_output = Path("..") / "output" / "gpt-4-api" / data_file_name.replace('.txt', '.relacje_gpt4')
        with open(file_output, 'w', encoding='utf-8') as f:
            f.write(output + '\n')

        # koszty
        tokens_in_data = count_tokens(text_from_file, "gpt2")
        tokens_in_result = count_tokens(output, "gpt2")
        # cena gpt-4 w openai 0.03$ za prompt, 0.06$ za wygenerowaną odpowiedź
        cena = ((tokens_in_data/1000) * 0.03) + ((tokens_in_result/1000) * 0.06)
        print(f'Koszt: {cena:.2f}$')