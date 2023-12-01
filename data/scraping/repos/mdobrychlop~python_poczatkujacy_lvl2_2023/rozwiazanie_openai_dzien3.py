import PyPDF2
from openai import OpenAI
import pandas as pd
import os

klient = OpenAI(api_key='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')

def pierwsza_strona_pdf_na_tekst(sciezka_pliku_pdf):
    """
    Czyta plik PDF i zwraca zawartość pierwszej strony jako string.
    """
    with open(sciezka_pliku_pdf, 'rb') as plik_pdf:
        pdfreader = PyPDF2.PdfReader(plik_pdf)
        strona = pdfreader.pages[0]
        tekst_strony = strona.extract_text()

    return tekst_strony

def uruchom_openai(prompt):
    odpowiedz = klient.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Jesteś pomocnym asystentem."},
            {"role": "user", "content": prompt},
        ]
    )
    return odpowiedz.choices[0].message.content

def zbuduj_prompt(tresc_pierwszej_strony):
    """
    Funkcja ma zwracać prompt dla modelu OpenAI, który będzie zawierał
    wszelkie niezbędne instrukcje, oraz treść pierwszej strony pliku PDF.
    """
    instrukcje = """Dostaniesz treść pierwszej strony pliku PDF, który zawiera artykuł naukowy.
Twoim zadaniem jest wypisanie na tej podstawie listy Pythona, która zawiera następujące elementy:
1. Tytuł artykułu.
2. Listę nazwisk autorów.
3. Krótkie podsumowanie abstraktu. (1 lub 2 zdania.)
Nie wypisuj żadnego tekstu poza wynikową listą. Podsumowanie abstraktu musi być napisane po Polsku.
Struktura wynikowej listy:
["Tytuł artykułu", ["Nazwisko1", "Nazwisko2"], "Podsumowanie abstraktu"]
Treść pierwszej strony pliku PDF:\n"""

    prompt = instrukcje + tresc_pierwszej_strony

    return prompt

def uporzadkuj_dane_z_pdfa(sciezka_do_pdf):
    """
    Funkcja, wewnątrz której odbywa się komunikacja z modelem OpenAI.
    Funkcja ma zwracać listę danych wyczytanych z pierwszej strony
    pliku PDF. Lista ma zawierać: tytuł, listę nazwisk autorów, 
    oraz krótkie podsumowanie abstraktu.
    """
    pierwsza_strona = pierwsza_strona_pdf_na_tekst(sciezka_do_pdf)
    prompt = zbuduj_prompt(pierwsza_strona)
    wynik = uruchom_openai(prompt)
    return wynik


lista_pdfow = ['pdfy/' + f for f in os.listdir('pdfy')]

lista_danych = []

for p in lista_pdfow:
    print(p)
    dane = uporzadkuj_dane_z_pdfa(p)
    print(dane)
    try:
        lista_danych.append(eval(dane))
    except:
        lista_danych.append(["NA", "NA", "NA"])

# zapisz dane do pliku excela
df = pd.DataFrame(lista_danych, columns=['Tytuł', 'Autorzy', 'Podsumowanie'])
df.to_excel('dane_z_pdfow.xlsx', index=False)


