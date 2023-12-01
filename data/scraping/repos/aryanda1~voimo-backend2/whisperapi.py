import openai
import os

def saveLyrics(mp3_path,output_dir,api_key):
    openai.api_key = api_key
    audio_file= open(mp3_path, "rb")
    transcript = openai.Audio.translate("whisper-1", audio_file)['text']
    text_file = open(os.path.join(output_dir,'vocals.txt'), "w")
    text_file.write("%s" % transcript)
    text_file.close()