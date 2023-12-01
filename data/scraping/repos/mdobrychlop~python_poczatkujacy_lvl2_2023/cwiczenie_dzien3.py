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
    pass

def uporzadkuj_dane_z_pdfa(sciezka_do_pdf):
    """
    Funkcja, wewnątrz której odbywa się komunikacja z modelem OpenAI.
    Funkcja ma zwracać listę danych wyczytanych z pierwszej strony
    pliku PDF. Lista ma zawierać: tytuł, listę nazwisk autorów, 
    oraz krótkie podsumowanie abstraktu.
    """
    pass


lista_pdfow = ['pdfy/' + f for f in os.listdir('pdfy')]
print(lista_pdfow)



