# Biblioteka Langchain. Szablony oraz połączenie z OpenAI,

from langchain.llms import OpenAI

if __name__ == '__main__':
    with open('api.key', 'r') as openai_api_key:
        openai_api_key = openai_api_key.read().strip()

    openai = OpenAI(
        model_name="text-davinci-003",
        openai_api_key=openai_api_key
    )

    review = "Jeśli  ktoś  lubi  zatłoczone hałaśliwe miejsca to może  w lipcu  się  tam wybrać, mi to zupełnie  nie odpowiada,  a wręcz jest męczące, ciężko  przemieszczać się po promenadzie a wejściówki  na tak zatłoczone  molo  dla  4 osobowej  rodziny  to i tak za drogie. "

    prompt = f"""Opinia użytkownika: {review}

    Czy ta opinia jest pozytywna, czy negatywna?"""

    print(openai(prompt, max_tokens=50, temperature=0.0,))
