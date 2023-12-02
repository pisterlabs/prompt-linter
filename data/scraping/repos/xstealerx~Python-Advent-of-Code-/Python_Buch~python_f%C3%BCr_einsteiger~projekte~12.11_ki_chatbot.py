import openai 

API_KEY = "DEIN-API-KEY"

def fragen(frage):
  ergebnis = openai.Completion.create(
     model="text-davinci-003", 
     prompt=frage,
     max_tokens=2048,
     api_key=API_KEY
  )
  antwort = ergebnis.choices[0].text
  return antwort

if __name__ == "__main__":
  print("Komm, lass uns chatten!")
  while (frage := input("\n> ")) != "X":
    antwort = fragen(frage)
    print(antwort)
