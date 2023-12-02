
"""
Whisper: es un modelo
de nlp para audio
"""

# Libreria

import openai

# Datos

openai.api_key = "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"

# Transcribir audio

url = "C:/Users/Angelica Gerrero/Videos/Audio_Sarcastico.wav"

with open(url, "rb") as audio:
	transcripcion = openai.Audio.transcribe("whisper-1", audio)
	
transcripcion = transcripcion["text"]
print("Transcripcion audio-texto: \n\n", transcripcion)

# Traduccion de audio

url = open(url, "rb")

traduccion = openai.Audio.translate("whisper-1", url)
traduccion = traduccion["text"]
url.close()

print("\nTraduccion español-ingles: \n\n", traduccion)