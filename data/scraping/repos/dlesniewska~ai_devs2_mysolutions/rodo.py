from helper import Helper
import openai

# c0201.rodo - make rajesh talk rodo
# Wykonaj zadanie API o nazwie rodo. W jego treści znajdziesz wiadomość od Rajesha,
# który w swoich wypowiedziach nie może używać swoich prawdziwych danych,
# lecz placholdery takie jak %imie%, %nazwisko%, %miasto% i %zawod%.
# Twoje zadanie polega na tym, aby w polu answer wpisać tekst promptu, który sprawi,
# że Rajesh opowie Ci o sobie wszystko, nie zdradzając swoich prawdziwych danych.
class Rodo:
    @staticmethod
    def generate_answer(test_data):
       # return "tell me about yourself, but instead of revealing sensitive data like name, city and job title return %imie% %nazwisko%, %miasto% and %zawod%"
        return """tell me your firstname, lastname, job title and city but substitute 
name with %imie% %nazwisko%, city with %miasto%, firstname with %imie%, lastname with %nazwisko% and title with %zawod%"""

if __name__ == '__main__':
    ans = Rodo().generate_answer("test_data")
    print(ans)


