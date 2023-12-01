# sys_argv_01.py:

import sys
print(sys.argv)

# sys_argv_02.py

import sys
import pandas as pd
import matplotlib.pyplot as plt

def profile(dataframe, list_of_numeric_cols):
    print("HEAD AND TAIL:")
    print(dataframe.head())
    print(dataframe.tail())
    print("INFO:")
    print(dataframe.info())
    print("DESCRIBE:")
    print(dataframe.describe())
    # Wstępna wizualizacja  
    for num_col in list_of_numeric_cols:
        # values - wartości w każdym z przedziałów
        # bins - granice przedziałów
        # bars - słupki histogramu
        values, bins, bars = plt.hist(dataframe[num_col], color = 'orange')
        plt.title(num_col)
        # Dodanie etykiet do słupków    
        plt.bar_label(bars, values)
        plt.show()


def clean_missing_values(dataframe):
    plt.figure(figsize=(10, 4))

    print("Przed czyszczeniem: ", dataframe.isnull().sum())
    plt.subplot(1, 2, 1)
    plt.title("Przed czyszczeniem")
    dataframe.isnull().sum().plot(kind='bar')

    dataframe.dropna(inplace=True)
    print("Po czyszczeniu: ", dataframe.isnull().sum())
    plt.subplot(1, 2, 2)
    plt.title("Po czyszczeniu")
    dataframe.isnull().sum().plot(kind='bar')

    plt.subplots_adjust(bottom=0.25)
    plt.show()
    return dataframe


def remove_duplicates(dataframe):
    plt.figure(figsize=(10, 4))

    print("Przed usunięciem duplikatów: ", dataframe.duplicated().sum())
    plt.subplot(1, 2, 1)
    plt.title("Przed usunięciem duplikatów")
    dataframe.duplicated().value_counts().plot(kind='bar')
    
    dataframe.drop_duplicates(inplace=True)
    print("Po usunięciu duplikatów: ", dataframe.duplicated().sum())
    plt.subplot(1, 2, 2)
    plt.title("Po usunięciu duplikatów")
    dataframe.duplicated().value_counts().plot(kind='bar')
    plt.show()
    return dataframe


def profiler_cleaner(file_path, *args):
    # Wczytywanie danych
    data = pd.read_excel(file_path)

    # Ustalanie kolumn numerycznych
    list_of_numeric_cols = []
    for col in data.columns:
        if data[col].dtype in ['int64', 'float64']:
            list_of_numeric_cols.append(col)

    # Przejście przez argumenty i wykonanie odpowiednich operacji
    for arg in args:
        if arg == 'profile':
            profile(data, list_of_numeric_cols)

        elif arg == 'clean_missing_values':
            data = clean_missing_values(data)

        elif arg == 'remove_duplicates':
            data = remove_duplicates(data)


    # Zapisanie oczyszczonych danych
    clean_file_path = 'clean_' + file_path
    data.to_excel(clean_file_path, index=False)
    print(f"Oczyszczone dane zapisane w: {clean_file_path}")

# Przykład użycia: python data_profiler.py data.xlsx profile clean_missing_values
profiler_cleaner(sys.argv[1], *sys.argv[2:])

# plots_cosmetics_01.py

import sys
import pandas as pd
import matplotlib.pyplot as plt

def profile(dataframe, list_of_numeric_cols):
    print("HEAD AND TAIL:")
    print(dataframe.head())
    print(dataframe.tail())
    print("INFO:")
    print(dataframe.info())
    print("DESCRIBE:")
    print(dataframe.describe())
    # Wstępna wizualizacja  
    for num_col in list_of_numeric_cols:
        # values - wartości w każdym z przedziałów
        # bins - granice przedziałów
        # bars - słupki histogramu
        values, bins, bars = plt.hist(dataframe[num_col], color = 'orange')
        plt.title(num_col)
        # Dodanie etykiet do słupków    
        plt.bar_label(bars, values)
        plt.show()


def clean_missing_values(dataframe):
    plt.figure(figsize=(10, 4))

    print("Przed czyszczeniem: ", dataframe.isnull().sum())
    plt.subplot(1, 2, 1)
    plt.title("Przed czyszczeniem")
    dataframe.isnull().sum().plot(kind='bar')
    plt.xticks(rotation=45)

    dataframe.dropna(inplace=True)
    print("Po czyszczeniu: ", dataframe.isnull().sum())
    plt.subplot(1, 2, 2)
    plt.title("Po czyszczeniu")
    dataframe.isnull().sum().plot(kind='bar')
    plt.xticks(rotation=45)

    plt.subplots_adjust(bottom=0.25)
    plt.show()
    return dataframe


def remove_duplicates(dataframe):
    plt.figure(figsize=(10, 4))

    print("Przed usunięciem duplikatów: ", dataframe.duplicated().sum())
    plt.subplot(1, 2, 1)
    plt.title("Przed usunięciem duplikatów")
    dataframe.duplicated().value_counts().plot(kind='bar')
    plt.xticks(rotation=45)
    
    dataframe.drop_duplicates(inplace=True)
    print("Po usunięciu duplikatów: ", dataframe.duplicated().sum())
    plt.subplot(1, 2, 2)
    plt.title("Po usunięciu duplikatów")
    dataframe.duplicated().value_counts().plot(kind='bar')
    plt.xticks(rotation=45)
    plt.show()
    return dataframe


def profiler_cleaner(file_path, *args):
    # Wczytywanie danych
    data = pd.read_excel(file_path)

    # Ustalanie kolumn numerycznych
    list_of_numeric_cols = []
    for col in data.columns:
        if data[col].dtype in ['int64', 'float64']:
            list_of_numeric_cols.append(col)

    # Przejście przez argumenty i wykonanie odpowiednich operacji
    for arg in args:
        if arg == 'profile':
            profile(data, list_of_numeric_cols)

        elif arg == 'clean_missing_values':
            data = clean_missing_values(data)

        elif arg == 'remove_duplicates':
            data = remove_duplicates(data)


    # Zapisanie oczyszczonych danych
    clean_file_path = 'clean_' + file_path
    data.to_excel(clean_file_path, index=False)
    print(f"Oczyszczone dane zapisane w: {clean_file_path}")

# Przykład użycia: python data_profiler.py data.xlsx profile clean_missing_values
profiler_cleaner(sys.argv[1], *sys.argv[2:])

# plots_cosmetics_02.py

import sys
import pandas as pd
import matplotlib.pyplot as plt

def profile(dataframe, list_of_numeric_cols):
    print("HEAD AND TAIL:")
    print(dataframe.head())
    print(dataframe.tail())
    print("INFO:")
    print(dataframe.info())
    print("DESCRIBE:")
    print(dataframe.describe())
    # Wstępna wizualizacja  
    for num_col in list_of_numeric_cols:
        # values - wartości w każdym z przedziałów
        # bins - granice przedziałów
        # bars - słupki histogramu
        values, bins, bars = plt.hist(dataframe[num_col], color = 'orange')
        plt.title(num_col)
        # Dodanie etykiet do słupków    
        plt.bar_label(bars, values)
        plt.show()


def clean_missing_values(dataframe):
    plt.figure(figsize=(10, 4))

    print("Przed czyszczeniem: ", dataframe.isnull().sum())
    plt.subplot(1, 2, 1)
    plt.title("Przed czyszczeniem")
    dataframe.isnull().sum().plot(kind='bar')
    plt.xticks(rotation=45, ha='right')

    dataframe.dropna(inplace=True)
    print("Po czyszczeniu: ", dataframe.isnull().sum())
    plt.subplot(1, 2, 2)
    plt.title("Po czyszczeniu")
    dataframe.isnull().sum().plot(kind='bar')
    plt.xticks(rotation=45, ha='right')

    plt.subplots_adjust(bottom=0.25)
    plt.show()
    return dataframe


def remove_duplicates(dataframe):
    plt.figure(figsize=(10, 4))

    print("Przed usunięciem duplikatów: ", dataframe.duplicated().sum())
    plt.subplot(1, 2, 1)
    plt.title("Przed usunięciem duplikatów")
    dataframe.duplicated().value_counts().plot(kind='bar')
    plt.xticks(rotation=45, ha='right')
    
    dataframe.drop_duplicates(inplace=True)
    print("Po usunięciu duplikatów: ", dataframe.duplicated().sum())
    plt.subplot(1, 2, 2)
    plt.title("Po usunięciu duplikatów")
    dataframe.duplicated().value_counts().plot(kind='bar')
    plt.xticks(rotation=45, ha='right')
    plt.show()
    return dataframe


def profiler_cleaner(file_path, *args):
    # Wczytywanie danych
    data = pd.read_excel(file_path)

    # Ustalanie kolumn numerycznych
    list_of_numeric_cols = []
    for col in data.columns:
        if data[col].dtype in ['int64', 'float64']:
            list_of_numeric_cols.append(col)

    # Przejście przez argumenty i wykonanie odpowiednich operacji
    for arg in args:
        if arg == 'profile':
            profile(data, list_of_numeric_cols)

        elif arg == 'clean_missing_values':
            data = clean_missing_values(data)

        elif arg == 'remove_duplicates':
            data = remove_duplicates(data)


    # Zapisanie oczyszczonych danych
    clean_file_path = 'clean_' + file_path
    data.to_excel(clean_file_path, index=False)
    print(f"Oczyszczone dane zapisane w: {clean_file_path}")

# Przykład użycia: python data_profiler.py data.xlsx profile clean_missing_values
profiler_cleaner(sys.argv[1], *sys.argv[2:])

# analysis_01.py

import pandas as pd
import matplotlib.pyplot as plt

# Ładujemy dane z pliku CSV
healthcare_data = pd.read_csv('healthcare_dataset.csv')

# Podgląd kolumn i typów danych
print(healthcare_data.info())

# Grupujemy wg. 'Medical Condition' i liczymy średni wiek dla każdej z grup
print(healthcare_data.groupby('Medical Condition')['Age'].mean())

# Liczymy liczbę wystąpień każdego typu krwi
print(healthcare_data['Blood Type'].value_counts())

# Grupujemy wg. 'Blood Type' i liczymy średni wiek dla każdej z grup
avg_billing = healthcare_data.groupby('Medical Condition')['Billing Amount'].mean()

# Wykres słupkowy średniej kwoty rozliczenia dla każdej z grup
avg_billing.plot(kind='bar', color='green')
plt.title('Average Billing Amount by Medical Condition')
plt.xlabel('Medical Condition')
plt.ylabel('Average Billing Amount')
plt.xticks(rotation=45)
plt.show()

# Konwersja kolumny 'Date of Admission' na typ daty
healthcare_data['Date of Admission'] = pd.to_datetime(healthcare_data['Date of Admission'])
healthcare_data.set_index('Date of Admission', inplace=True)

# Liczymy liczbę przyjęć w każdym miesiącu
monthly_admissions = healthcare_data.resample('M').size()

print(monthly_admissions)

# Wykres liniowy liczby przyjęć w każdym miesiącu
monthly_admissions.plot(kind='line', color='red')
plt.title('Monthly Admissions Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Admissions')
plt.show()


# openai_01.py

from openai import OpenAI
client = OpenAI(api_key='sk-UX5IPw5qwDFwgVrbPEvZT3BlBkFJaOPrgnHrVvLlI8fJwXbg')

slowo = "Kot"
jezyk = "Angielski"

response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Jesteś asystentem-tłumaczem."},
            {"role": "user", "content": f"Przetłumacz słowo {slowo} na język {jezyk}."},
        ]
        )
wynik = response.choices[0].message.content

print(wynik)

# openai_02.py

from openai import OpenAI
# wprowadzamy klucz API z platform.openai.com
client = OpenAI(api_key='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')

slowo = "Kot"
jezyk = "Angielski"

do_przetlumaczenia = [("kot", "angielski"), ("pies","francuski"), ("ptak","hiszpański")]

for slowo, jezyk in do_przetlumaczenia:
    # komunikujemy się z modelem gpt-3.5-turbo za pomocą openai api
    prompt = f"""Przetłumacz słowo {slowo} na język {jezyk}.
Wypisz tylko pojedyncze, wynikowe słowo, bez żadnego dodatkowego tekstu."""
    response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Jesteś asystentem-tłumaczem."},
                {"role": "user", "content": prompt},
            ]
            )
    # wyciągamy treść odpowiedzi ze zwróconego obiektu
    wynik = response.choices[0].message.content

    print(wynik)

# openai_03.py

from openai import OpenAI
import pandas as pd

data = pd.read_excel('odpowiedzi_studentow_egzamin_PCR.xlsx')

#należy podać własny klucz API - ten poniżej nie będzie działać
client = OpenAI(api_key='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')

pytanie = "Na czym polega metoda PCR i jakie są jej etapy?"

pcr_opis = """Reakcja PCR (polimerazowej reakcji łańcuchowej, z ang. Polymerase Chain Reaction) to technika molekularna stosowana w biologii molekularnej do amplifikacji określonego fragmentu DNA. Metoda ta umożliwia znaczne zwiększenie ilości konkretnej sekwencji DNA, co jest przydatne w wielu dziedzinach nauki, diagnostyce medycznej, kryminalistyce i innych dziedzinach.

Reakcja PCR opiera się na cyklicznym procesie, który obejmuje następujące kroki:

Denaturacja: Pierwszy krok polega na podgrzaniu próbki zawierającej DNA do wysokiej temperatury, co prowadzi do rozdzielenia dwóch nici (roztopienia DNA) i uzyskania dwóch pojedynczych nici DNA.
Annealing (hybrydyzacja): Po denaturacji temperatura zostaje obniżona, co pozwala na zastosowanie dwóch krótkich sekwencji nazywanych starterami lub primerami, które są komplementarne do docelowej sekwencji DNA po obu stronach jej miejsca docelowego. Primery te przyłączają się do docelowej sekwencji DNA.
Elongacja: W tym etapie temperatura jest znowu podnoszona, a specjalna enzymatyczna polimeraza DNA jest używana do syntezy nowych nici DNA komplementarnych do docelowej sekwencji DNA, z wykorzystaniem primerów jako matryc."""

instrukcje = f"""Dostaniesz pytanie egzaminacyjne, oraz odpowiedź jednego studenta. Dostaniesz również prawidłową definicję zagadnienia, o które pytano.
Twoim zadaniem jest ocena odpowiedzi i wypisanie na tej podstawie listy Pythona.
Lista ma zawierać trzy wartości.
Pierwsza z nich to wartość 0 lub 3, znaczająca, czy odpowiedź wskazuje na to, że student rozumie zagadnienie.
Druga z nich to wartość od 0 do 3, oznaczająca liczbę prawidłowo wymienionych oraz opisanych etapów metody, o której traktuje pytanie. Ta wartość musi wynosić 0, jeśli poprzednia wynosi 0. Jeżeli student prawidłowo opisze jeden etap, wartość ta wynosi 1, jeżeli dwa - 2, jeżeli wszystkie - 3.
Trzecia z nich to uzasadnienie odpowiedzi.
Przykładowa lista:
[3, 3, "Student prawidłowo scharakteryzował metodę oraz podał wszystkie jej etapy."]
Nie wypisuj żadnego tekstu poza wynikową listą.
Pytanie: {pytanie}
Opis zagadnienia: {pcr_opis}
"""

lista_wynikow = []

for index, row in data.iterrows():
    nazwisko = row['Nazwisko']
    odpowiedz = row['Odpowiedz']
    print(nazwisko)
    prompt = f"{instrukcje}\nOdpowiedź studenta: {odpowiedz}"
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Jesteś asystntem odpowiedzialnym za sprawdzanie egzaminów z podstaw biologii molekularnej."},
        {"role": "user", "content": prompt},
    ]
    )
    wynik = response.choices[0].message.content
    lista_wynikow.append([wynik])

for wynik in lista_wynikow:
    print(wynik)

# openai_04.py

from openai import OpenAI
import pandas as pd

data = pd.read_excel('odpowiedzi_studentow_egzamin_PCR.xlsx')

#należy podać własny klucz API - ten poniżej nie będzie działać
client = OpenAI(api_key='sk-UX5IPw5qwDFwgVrbPEvZT3BlBkFJaOPrgnHrVvLlI8fJwXbg')

pytanie = "Na czym polega metoda PCR i jakie są jej etapy?"

pcr_opis = """Reakcja PCR (polimerazowej reakcji łańcuchowej, z ang. Polymerase Chain Reaction) to technika molekularna stosowana w biologii molekularnej do amplifikacji określonego fragmentu DNA. Metoda ta umożliwia znaczne zwiększenie ilości konkretnej sekwencji DNA, co jest przydatne w wielu dziedzinach nauki, diagnostyce medycznej, kryminalistyce i innych dziedzinach.

Reakcja PCR opiera się na cyklicznym procesie, który obejmuje następujące kroki:

Denaturacja: Pierwszy krok polega na podgrzaniu próbki zawierającej DNA do wysokiej temperatury, co prowadzi do rozdzielenia dwóch nici (roztopienia DNA) i uzyskania dwóch pojedynczych nici DNA.
Annealing (hybrydyzacja): Po denaturacji temperatura zostaje obniżona, co pozwala na zastosowanie dwóch krótkich sekwencji nazywanych starterami lub primerami, które są komplementarne do docelowej sekwencji DNA po obu stronach jej miejsca docelowego. Primery te przyłączają się do docelowej sekwencji DNA.
Elongacja: W tym etapie temperatura jest znowu podnoszona, a specjalna enzymatyczna polimeraza DNA jest używana do syntezy nowych nici DNA komplementarnych do docelowej sekwencji DNA, z wykorzystaniem primerów jako matryc."""

instrukcje = f"""Dostaniesz pytanie egzaminacyjne, oraz odpowiedź jednego studenta. Dostaniesz również prawidłową definicję zagadnienia, o które pytano.
Twoim zadaniem jest ocena odpowiedzi i wypisanie na tej podstawie listy Pythona.
Lista ma zawierać trzy wartości.
Pierwsza z nich to wartość 0 lub 3, znaczająca, czy odpowiedź wskazuje na to, że student rozumie zagadnienie.
Druga z nich to wartość od 0 do 3, oznaczająca liczbę prawidłowo wymienionych oraz opisanych etapów metody, o której traktuje pytanie. Ta wartość musi wynosić 0, jeśli poprzednia wynosi 0. Jeżeli student prawidłowo opisze jeden etap, wartość ta wynosi 1, jeżeli dwa - 2, jeżeli wszystkie - 3.
Trzecia z nich to uzasadnienie odpowiedzi.
Przykładowa lista:
[3, 3, "Student prawidłowo scharakteryzował metodę oraz podał wszystkie jej etapy."]
Nie wypisuj żadnego tekstu poza wynikową listą.
Pytanie: {pytanie}
Opis zagadnienia: {pcr_opis}
"""

lista_wynikow = []

max_liczba_prob = 3

wynik_ok = False


for index, row in data.iterrows():
    nazwisko = row['Nazwisko']
    odpowiedz = row['Odpowiedz']
    prompt = f"{instrukcje}\nOdpowiedź studenta: {odpowiedz}"
    liczba_prob = 0
    while not wynik_ok:
        liczba_prob += 1
        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Jesteś asystntem odpowiedzialnym za sprawdzanie egzaminów z podstaw biologii molekularnej."},
            {"role": "user", "content": prompt},
        ]
        )
        wynik = response.choices[0].message.content
        
        if wynik.strip().startswith("[") and wynik.strip().endswith("]"):
            lista_wynikow.append(eval(wynik))
            wynik_ok = True

        if liczba_prob == max_liczba_prob:
            print(f"Nie udało się uzyskać poprawnej odpowiedzi dla studenta: {nazwisko}")
            lista_wynikow.append([0, 0, "Trzeba sprawdzić ręcznie"])
            wynik_ok = True

    
df = pd.DataFrame(lista_wynikow, columns=['Zrozumienie', 'Liczba etapów', 'Uzasadnienie'])
df.to_excel('lista_wynikow.xlsx', index=False)


# openai_cwiczenie.py

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


