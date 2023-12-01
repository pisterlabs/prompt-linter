#pip3 install openai

import openai
import moviepy.editor as mp #for mov to mp3 conversion

openai.api_key = "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"

input_mov_file = "raw_mom.mov"
output_mp3_file = "raw_mom.mp3"
video_clip = mp.VideoFileClip(input_mov_file)
audio_clip = video_clip.audio
audio_clip.write_audiofile(output_mp3_file, codec="mp3")
video_clip.close()
audio_clip.close()

print(f"Conversion to MP3 complete. MP3 file saved as '{output_mp3_file}'.")


#using whisper
audio_file= open(output_mp3_file, "rb")
transcript = (openai.Audio.transcribe("whisper-1", audio_file)).text

file_path = "transcript_mom.txt"
with open(file_path, 'w') as file:
    file.write(transcript)