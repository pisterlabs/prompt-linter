from openai import OpenAI
import datetime

# Praktyczne zastosowanie Pythona w naukach przyrodniczych dla początkujących – poziom 2
# MATERIAŁ DODATKOWY (BONUS)
# Prosty "wrapper" dla OpenAI API, dzięki któremu możliwe jest prowadzenie rozmowy
# z wybranym modelem OpenAI, zgodnie z modelem opłat za liczbę tokenów, a nie
# za abonament ChatGPT Plus.

# Skrypt nie jest zabezpieczony przed przekroczeniem limitu tokenów w kontekście.
# Przekroczenie limitu będzie skutkowało zakończeniem rozmowy (błędem).

# definiujemy klucz OpenAI - klucz poniżej trzeba zastąpić własnym kluczem
klient = OpenAI(api_key='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')

def run_openai(messages_history, model_engine):
    odpowiedz = klient.chat.completions.create(
        model=model_engine,
        messages=messages_history,
        temperature=0.7,
    )
    return odpowiedz.choices[0].message.content


if __name__ == '__main__':
    # definiujemy domyślny model rozmowy (do zmiany w trakcie rozmowy)
    model_engine = "gpt-3.5-turbo"
    # rozpoczynamy listę wiadomości od wiadomości systemowej
    messages_history = [{"role": "system", "content": "You are a helpful assistant"},]

    # początkowy komunikat w konsoli dla użytkownika
    print("Welcome to the chatbot. Type 'exit' to quit, 'save' to save chat history to file, 'fromcode' to send message from code, 'fromfile' to send message from file, or model name to change model.")
    message = ""

    # skrypt działa w pętli tak długo, aż użytkownik nie wpisze "exit", lub program nie napotka na błąd
    while True:
        message = input("You: ")
        # po wpisaniu "exit" kończymy rozmowę
        if message == "exit":
            break
        # możemy zmieniać modele w trakcie rozmowy wpisując ich nazwy
        # historia rozmowy pozostaje niezmieniona
        elif message == "gpt-4":
            model_engine = "gpt-4"
            continue
        elif message == "gpt-3.5-turbo":
            model_engine = "gpt-3.5-turbo"
            continue
        elif message == "gpt-4-turbo":
            model_engine = "gpt-4-turbo"
            continue

        # możemy wpisać waidomość w kodzie poniżej, a potem wysłać ją automatycznie wpisując "fromcode"
        elif message == "fromcode":
            message = """Test"""

        # zapisujemy historię rozmowy do pliku tekstowego (w innym wypadku rozmowa przepada po zamknięciu terminala)
        elif message == "save":          
            now = datetime.datetime.now()
            filename = "chat_history_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
            with open(filename, "w") as f:
                for item in messages_history:
                    f.write("%s\n" % item)
            print("Chat history saved to file: " + filename)
            continue
        
        # możemy też wczytać wiadomość z pliku o nazwie "input.txt"
        elif message == "fromfile":
            filename = "input.txt"
            with open(filename, "r") as f:
                message = f.read()

            
        # dodajemy wiadomosć do listy wiadomości
        messages_history.append({"role": "user", "content": message})

        # całą listę wiadomości wysyłamy do OpenAI API
        chatbot_response = run_openai(messages_history, model_engine)

        # wyświetlamy odpowiedź chatbota
        print("Assistant: " + chatbot_response)

        # dodajemy odpowiedź chatbota do listy wiadomości
        messages_history.append({"role": "assistant", "content": chatbot_response})
