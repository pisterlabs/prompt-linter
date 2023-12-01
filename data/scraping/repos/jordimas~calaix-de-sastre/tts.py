from pathlib import Path
from openai import OpenAI
client = OpenAI()


voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

for voice in voices:

    speech_file_path = Path(__file__).parent / f"speech-{voice}.mp3"
    response = client.audio.speech.create(
      model="tts-1-hd",
      voice=voice,
      input="Softcatalà fa vint-i-cinc anys, i per a celebrar-ho hem organitzat actes arreu del territori amb l’objectiu de trobar-nos amb companys, col·laboradors, antics membres i altres persones que ens donen suport. Volem recordar i repassar la nostra trajectòria, analitzar la situació actual de la llengua i debatre els reptes de futur. Els diversos actes tindran una estructura similar, tot i que comptaran amb diferents conduccions i persones convidades."
    )

    response.stream_to_file(speech_file_path)
