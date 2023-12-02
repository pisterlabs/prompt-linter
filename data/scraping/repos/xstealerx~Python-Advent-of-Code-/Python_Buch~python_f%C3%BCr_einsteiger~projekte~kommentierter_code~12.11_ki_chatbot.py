import openai 
# API-Key, den man für die OpenAI API benötigt.
API_KEY = "DEIN-API-KEY"
# Funktion zum Fragenstellen.
def fragen(frage):
  ergebnis = openai.Completion.create(
     model="text-davinci-003",  # Sprachmodell
     prompt=frage,              # Frage an den Chatbot
     max_tokens=2048,           # Anzahl an Tokens
     api_key=API_KEY            # API-Key
  )
  antwort = ergebnis.choices[0].text  # Antwort auslesen
  return antwort                      # Antwort zurückgeben
# Hautprogramm
if __name__ == "__main__":
  # Chat starten.
  print("Komm, lass uns chatten!")
  # Benutzereingaben abfragen, bis der Benutzer ein "X" eingibt.
  while (frage := input("\n> ")) != "X":
    # Sprachmodell eine Frage stellen und Antwort speichern
    antwort = fragen(frage)
    # Antwort ausgeben
    print(antwort)
