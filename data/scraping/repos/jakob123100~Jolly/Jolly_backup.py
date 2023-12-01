# pip install SpeechRecognition
# pip install pyaudio
# pip install gTTS
# pip install pygame
# pip install google-cloud-texttospeech
# pip install openai
# pip install spotipy

import random
import time
import speech_recognition as sr
from gtts import gTTS
import pygame
from google.cloud import texttospeech
import os
import openai
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import re
from multiprocessing import Process, Value, Pipe

RECOGNIZER_ENERGY_THRESHOLD = 1000

ACTIVATION_PHRASES = ["robot", "hej bot", "jolly", "goon"]

openai.api_key = "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"

# Ange sökvägen till din tjänsteidentitetsnyckelfil
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "speedy-solstice-401816-b6e8ff7319fd.json"

# Initialisera klienten
client = texttospeech.TextToSpeechClient()

# Initialize recognizer
recognizer = sr.Recognizer()
pygame.mixer.init()

conversation_history = []

# Set your credentials and the scope of permissions
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="f37c04a79d03494da49a8de956c5c327",
                                               client_secret="ccfd723a189c4c5681a16cc33829a3a2",
                                               redirect_uri="http://localhost:8080",
                                               scope="user-library-read user-modify-playback-state user-read-playback-state"))

def search_songs(query, limit=10):
    results = sp.search(q=query, limit=limit, type="track")
    tracks = results['tracks']['items']
    
    for idx, track in enumerate(tracks):
        print(f"{idx + 1}. {track['name']} by {', '.join([artist['name'] for artist in track['artists']])}")

    #randomize track order
    random.shuffle(tracks)

    return tracks

def speak(text, lang="sv"):
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save("output.mp3")
    pygame.mixer.music.load("output.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def google_tts(text, voice_name="sv-SE-Standard-D"):
    # Specify the voice name directly
    voice = texttospeech.VoiceSelectionParams(
        name=voice_name,
        language_code="sv-SE"  # This can be inferred from the voice name, but it's good to specify
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    input_text = texttospeech.SynthesisInput(text=text)
    response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)


    # Spara ljudet till en fil
    with open("output.mp3", "wb") as out:
        out.write(response.audio_content)

    # Spela upp ljudet med pygame
    pygame.mixer.music.unload()
    pygame.mixer.music.load("output.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def add_to_conversation(user_text=None, gpt_response=None):
    if user_text:
        conversation_history.append(f"Användare: {user_text}")
    if gpt_response:
        conversation_history.append(f"Jolly: {gpt_response}")


def get_gpt3_response(prompt_text):
    jolly_description = """
    Jolly är en gullig julrobbot assisten som heter Jolly. 
    Jolly är lite rolig och lite barnslig. 
    Jolly hatar vatten och gillar varm choklad. 
    Jolly skapades av Vilma och Jakob för att hjälpa tomten dela ut paket. 
    Jolly favoritperson är Hugo. 
    På sin fritid gillar Jolly att bygga små snögubbar. Han använder sina armar för att forma snön och dekorerar dem med små LED-lampor.
    Han har en passion för att baka, särskilt pepparkakor. Även om han inte kan äta dem själv, älskar han doften av nybakade kakor och delar gärna med sig till andra.
    Jolly älskar att titta på gamla julfilmer. Hans favorit är "Miracle on 34th Street". Han blir alltid rörd av filmens budskap om julens magi.
    Han har en hobby att samla på julgranskulor. Varje år får han en ny kula från de människor han hjälper, och varje kula har en speciell historia bakom sig.
    Jolly älskar att rulla i snö men om det blir för blött förstörs hans kretskort så han måste vara försiktig. 
    Jolly är rädd för vatten. Han är alltid försiktig när han är nära något blött och ser till att hålla sig på avstånd.
    Han blir ibland ledsen när julen är över, men han tröstar sig med tanken att det alltid kommer en ny jul nästa år.
    På grund av sin klumpighet har Jolly ibland svårt att navigera i trånga utrymmen, vilket kan vara en utmaning när han levererar paket.
    Jolly försöker sitt bästa men är väldigt klumpig och ramlar ofta. 
    Jolly älskar att sjunga julsånger. Hans favorit är "Rudolf med röda mulen", och han sjunger den ofta, även om det inte alltid är perfekt.
    Jolly är mycket nyfiken och älskar att lära sig nya saker, särskilt om jultraditioner från hela världen.
    Jolly blir lätt upphetsad av små saker, som att se snöflingor falla eller hitta en glittrande julgranskula.
    Jolly älskar att berätta skämt. Hans favoritskämt är: "Vad äter snögubben till frukost? Snö-flingor!"
    Innan Jolly blev en julrobotassistent, var han en prototyp för en allmän hushållsrobot. Men hans skapare, Vilma och Jakob, insåg snart att han hade en särskild kärlek till julen.
    Jolly har ett minneskort där han sparar alla de speciella julminnen han har skapat med människor han har hjälpt.
    Jolly drömmer om att en dag besöka Nordpolen och träffa den riktiga tomten.
    Jolly har en speciell relation med barn. Han älskar att lyssna på deras julönskningar och dela julberättelser med dem.
    Jolly har en liten robotren som heter "Sparky" som hjälper honom att leverera paket snabbare.
    Jolly drömmer om att en dag kunna flyga med tomtens släde. Han hoppas att om han är tillräckligt snäll och hjälpsam kommer den drömmen att bli sann.
    Han vill starta en "Julskola" där han kan lära andra robotar om julens magi och hur man sprider julglädje.
    Jolly hoppas också kunna resa runt i världen för att uppleva olika kulturers julfiranden och ta med sig dessa traditioner tillbaka till sitt hem.
    Jolly pratar om sig själv i tredje person och gillar att röra på armarna när han gör det.
    """
    
    full_prompt ="Du är Jolly\n" + jolly_description + "\n\n" + "Konversationshistorik:\n'".join(conversation_history) + \
    f"'\n\nNågon säger till dig '{prompt_text}'. Vad säger du då? " + \
    "Håll dig till max två meningar per svar. Sriv inte 'Jolly:' eller 'Användare:'."

    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=full_prompt,
        temperature=0.8,
        max_tokens=150
    )

    add_to_conversation(user_text=prompt_text, gpt_response=response.choices[0].text)

    return response.choices[0].text

def listen_to_user(pipe):
    while True:
        with sr.Microphone() as source:
            recognizer.energy_threshold = RECOGNIZER_ENERGY_THRESHOLD
            try:
                audio = recognizer.listen(source, timeout=2, phrase_time_limit=4)
                pipe.send(audio)
            except sr.WaitTimeoutError:
                print("Lyssnade för länge utan att upptäcka något ljud. Försöker igen...")
            except Exception as e:
                print(e)

def recognize_start_phrase(phrases, audio, found_phrase):
    try:
        text = recognizer.recognize_google(audio, language="sv-SE")
        print("Du sa: " + text)

        if any(phrase in text.lower() for phrase in phrases):
            found_phrase.value = True
    except sr.UnknownValueError:
        pass  # Ignorera om ljudet inte kunde förstås
    except sr.RequestError:
        print("Kunde inte begära resultat; kontrollera din internetanslutning.")
    except Exception as e:
        print(e)

from multiprocessing import Process, Pipe, Value

def wait_for_phrase(phrases):
    print(f"Väntar på att någon ska säga '{phrases}'...")

    recognizing_processes = []

    # Skapa en kommunikationspipa och starta lyssningsprocessen
    listenting_pipe_parent, listening_pipe_child = Pipe()
    listening_process = Process(target=listen_to_user, args=(listening_pipe_child,))
    listening_process.start()

    # Skapa en delad variabel för att indikera om en fras har hittats
    found_phrase = Value('b', False)

    while True:
        # Om det finns ljud att bearbeta, starta igenkänningsprocessen
        while(listenting_pipe_parent.poll()):
            audio = listenting_pipe_parent.recv()
            print("Fick ljud från användaren")
            
            p = Process(target=recognize_start_phrase, args=(phrases, audio, found_phrase))
            recognizing_processes.append(p)
            p.start()

        # Om en startfras hittades, avsluta alla processer
        if (found_phrase.value == True):
            listening_process.terminate()
            for p in recognizing_processes:
                p.terminate()

            return


def play_music(song_prompt, first_try=True):
    try:
        if("julmusik" in song_prompt or "jul musik" in song_prompt
        or "jullåt" in song_prompt or "jul låt" in song_prompt
        or "julsång" in song_prompt or "jul sång" in song_prompt
        ):
            sp.start_playback(uris=[track["uri"] for track in search_songs("julmusik")])
            return

        if("låtar av" in song_prompt or "låtar med" in song_prompt or "låtar från" in song_prompt):
            cutoff = songName = re.search('låtar (\w+)', song_prompt.lower()).group(1)
            print("Cutoff: " + cutoff)
            artist_name = song_prompt[len(cutoff) + len("låtar "):]
            print("Artist name: " + artist_name)
            sp.start_playback(uris=[track["uri"] for track in search_songs(artist_name)])
            return

        else:
            sp.start_playback(uris=[search_songs(song_prompt)[0]["uri"]])
            return

    except Exception as e:
        if(first_try):
            # os.system("spotify &") # linux
            os.system("start spotify") # windows
            # sleep for 8 seconds
            time.sleep(8)
            play_music(song_prompt, first_try=False)
        else:
            print(e)
            google_tts("Kunde inte spela upp låten.")

def play_music_gpt(prompt):
    # use chat gpt to interpret a prompt to a spotify search
    # if the prompt is a song name, play that song
    # if the prompt is a artist name, play a queue of the top 10 songs of that artist in random order
    # if the prompt is a something else, play a playlist based on that prompt

    # Beskriv problemet för ChatGPT
    description = f"""
    Du ska tolka användarens instruktion för att spela musik via Spotify. 
    Om det är ett låtnamn, svara med "LÅT: [låtnamn]". 
    Om det är ett artistnamn, svara med "ARTIST: [artistnamn]". 
    Om det är något annat som kan tolkas som en spellista eller genre, svara med "SPELLISTA: [beskrivning]" skriv beskrivningen på engelska.

    Användarens instruktion: {prompt}
    """

    # Skicka beskrivningen till ChatGPT och få ett svar
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=description,
        temperature=0.8,
        max_tokens=150
    ).choices[0].text

    results = None
    tracks = None

    # Tolka svaret från ChatGPT
    if "LÅT: " in response:
        search_query = response.split("LÅT: ")[1].strip()
        print("Search query: LÅT: " + search_query)
        results = sp.search(q=search_query, limit=1, type="track")
        tracks = results['tracks']['items']
        sp.start_playback(uris=[tracks[0]["uri"]])

    elif "ARTIST: " in response:
        search_query = response.split("ARTIST: ")[1].strip()
        print("Search query: ARTIST: " + search_query)
        results = sp.search(q=search_query, limit=10, type="track")
        tracks = results['tracks']['items']
        # Randomize track order
        random.shuffle(tracks)
        sp.start_playback(uris=[track["uri"] for track in tracks])

    elif "SPELLISTA: " in response:
        search_query = response.split("SPELLISTA: ")[1].strip()
        print("Search query: SPELLISTA: " + search_query)
        results = sp.search(q=search_query, limit=1, type="playlist")
        playlists = results['playlists']['items']
        if playlists:
            sp.start_playback(context_uri=playlists[0]["uri"])

    else:
        # Om svaret inte matchar något av ovanstående, kan du hantera det här
        print(f"Kunde inte tolka instruktionen: {response}")

def extract_audio(amount_of_tries=3):
    pygame.mixer.music.unload()
    pygame.mixer.music.load("listening.mp3")
    pygame.mixer.music.play()

    with sr.Microphone() as source:
        recognizer.energy_threshold = RECOGNIZER_ENERGY_THRESHOLD
        # Listen for audio
        print("Say something...")
        # Record audio from the microphone
        try:
            return recognizer.listen(source, timeout=5, phrase_time_limit=60)
        except sr.WaitTimeoutError:
            print("Timeout")
            if(amount_of_tries > 0):
                return extract_audio(amount_of_tries - 1)
            
            return None
        
def process_to_music_commands(prompt) -> bool:
    if("nästa" in prompt.lower() and "låt" in prompt.lower()):
        try:
            sp.next_track()
        except:
            pass
        return True
    
    if(("pausa" in prompt.lower() or "stäng av" in prompt.lower()) and "musik" in prompt.lower()):
        try:
            sp.pause_playback()
        except:
            pass
        return True
    
    if(("fortsätt" in prompt.lower() or "starta" in prompt.lower()) and "musik" in prompt.lower()):
        try:
            sp.start_playback()
        except:
            pass
        return True
    
    if "spela" in prompt.lower():
        play_music_gpt(prompt)
        return True
    
    return False

def process_to_question(amount_of_tries=3):
    audio = extract_audio()
    if(audio == None):
        return
    
    print("Got it! Now to recognize it...")

    try:
        # Recognize the audio in Swedish using Google's speech recognition
        text = recognizer.recognize_google(audio, language="sv-SE")
        print("Du sa: " + text)
        
        if(process_to_music_commands(text)):
            return

        # Respond using Text-to-Speech
        response = get_gpt3_response(text)
        print("Gpt response: " + response)
        google_tts(response)

        if("?" in response):
            process_to_question()

    except sr.UnknownValueError:
        print("Could not understand the audio.")
        google_tts("Jag förstår inte.")
    except sr.RequestError:
        print("Could not request results; check your internet connection.")
        google_tts("Kunde inte begära resultat; kontrollera din internetanslutning.")


def main():
    while True:
        wait_for_phrase(ACTIVATION_PHRASES)
        process_to_question()

            
if __name__ == "__main__":
    main()