import asyncio
import re
import openai
import pyttsx3
import speech_recognition as sr 
import time
from EdgeGPT import Chatbot, ConversationStyle

#mise en place des clef OpenAI API
openai.api_key = "..."

# Initialisation du module vocal
engine = pyttsx3.init()

def tatt(filenam):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filenam) as source:
        audio = recognizer.record(source)
    try : 
        return recognizer.recognize_google(audio, language="fr-FR")
    except :
        print('Skipping unknown error')

def generate_response(prompt):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=4000,
        n=1,
        stop=None,
        temperature=0.5
    )
    return response["choices"][0]["text"]

async def generate_IA(prompt):
    speak_text('Jake réfléchit la réponse arrive bientôt')
    print("Question reçu ...")
    speak_text("Question reçu")
    bot = await Chatbot.create()
    speak_text("En cour de traitement ...")
    response = await bot.ask(prompt=prompt, conversation_style=ConversationStyle.creative)

    print("Question Traité ...")
    speak_text("Question Traité")

    # Sélection de la réponse du bot
    bot_response = None
    for message in response["item"]["messages"]:
        if message["author"] == "bot":
            bot_response = message["text"]

    print("Génération de la réponse ...")
    speak_text("Génération de la réponse")

    if bot_response:
       # Supprime les balises dans la réponse
        bot_response = re.sub(r'\[.*?\]', '', bot_response)
        bot_response = re.sub(r'Bing', 'Jake', bot_response)
        print(f"Jake : {bot_response}")
        speak_text(bot_response)

    await bot.close()

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def main():
    while True:
        print("Dite 'Jake' pour commencer à enregistrer votre requête ...")
        with sr.Microphone() as source :
            recognizer = sr.Recognizer()
            audio = recognizer.listen(source)
            try :
                transcription = recognizer.recognize_google(audio, language="fr-FR")
                if transcription.lower() == "jake":
                    filename = "input.wav"
                    speak_text("Jake vous écoute quel est votre question ?")
                    print("Posez votre question ...")
                    with sr.Microphone() as source :
                        recognizer = sr.Recognizer()
                        source.pausetreshold = 1
                        audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)
                        with open(filename, "wb") as f :
                            f.write(audio.get_wav_data())
                    
                    # Retranscription de la requête
                    text = tatt(filename)
                    if text :
                        print(f"Vous venez de dire: \n{text}")

                        #génération de la réponce
                        if text == 'est-ce que tu me comprends':
                            speak_text("Oui mais je ne suis pas encore relier à une IA")

                        elif text == 'au revoir Jake' :
                            speak_text("Au revoir au plaisir de vous entendre à nouveau monsieur")
                            print("Merci d'avoire utiliser le programme")
                            break

                        asyncio.run(generate_IA(text))
                        
            except Exception as e:
                print(f"An error occured: {e}")


if __name__ == "__main__":
    main()