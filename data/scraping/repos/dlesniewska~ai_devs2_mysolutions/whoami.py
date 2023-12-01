import requests
from requests import Response

from helper import Helper
import openai
from authorization import AuthorizationTokenResolver
from testDataResolver import TestDataResolver
from answersender import AnswerSender
# c0303.whoami - riddle me this game with multiple hints
# Rozwiąż zadanie o nazwie “whoami”. Za każdym razem, gdy pobierzesz zadanie,
# system zwróci Ci jedną ciekawostkę na temat pewnej osoby. Twoim zadaniem jest zbudowanie mechanizmu,
# który odgadnie, co to za osoba. W zadaniu chodzi o utrzymanie wątku w konwersacji z backendem. Jest to dodatkowo utrudnione przez fakt,
# że token ważny jest tylko 2 sekundy (trzeba go cyklicznie odświeżać!). Celem zadania jest napisania mechanizmu, który odpowiada,
# czy na podstawie otrzymanych hintów jest w stanie powiedzieć, czy wie, kim jest tajemnicza postać.
# Jeśli odpowiedź brzmi NIE, to pobierasz kolejną wskazówkę i doklejasz ją do bieżącego wątku.
# Jeśli odpowiedź brzmi TAK, to zgłaszasz ją do /answer/
# Wybraliśmy dość ‘ikoniczną’ postać, więc model powinien zgadnąć, o kogo chodzi, po maksymalnie 5-6 podpowiedziach.
# Zaprogramuj mechanizm tak, aby wysyłał dane do /answer/ tylko, gdy jest absolutnie pewny swojej odpowiedzi.
class Whoami:

    @staticmethod
    def generate_answer(test_data, token):
        openai.api_key = Helper().get_openapi_key()
        hint = test_data.json()["hint"]

        iteration = 0
        person = "NIE"
        while person == "NIE" and iteration < 6:
            print(f"Try nr...{iteration}")
            iteration += 1
            person = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "Answer the riddle: what is the name of the person from description. No comments or deviations. Only name only if you are absolutely sure or NIE if you are not sure. ###Description:###"},
                    {"role": "user",
                     "content": hint}
                ]).choices[0].message.content
            if person != "NIE":
                answer_sender = AnswerSender()
                answer_sender.give_answer(token, person)
                print("Answer was: ")
                print(person)
            else:
                #ask for another hint and new token
                authorization_token_resolver = AuthorizationTokenResolver()
                token = authorization_token_resolver.authorize("whoami")
                test_data_resolver = TestDataResolver()
                test_data = test_data_resolver.get_data(token)
                hint = hint + "." + test_data.json()["hint"]
                print("Hint is: ", hint)


if __name__ == '__main__':

    authorization_token_resolver = AuthorizationTokenResolver()
    token = authorization_token_resolver.authorize("whoami")
    test_data_resolver = TestDataResolver()
    test_data = test_data_resolver.get_data(token)
    print("Test data: ", test_data.json())

    # send result #before: answer = testData.json()[fieldName]
    Whoami.generate_answer(test_data, token)


# """blogger here
# ###
# rules:
# ###
# -i create a blog entry about pizza margherita containing 4 titles
# -list titles no deviation"""
