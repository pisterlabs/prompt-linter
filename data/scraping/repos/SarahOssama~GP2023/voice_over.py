import os
from django.conf import settings
from pydub import AudioSegment
import pyttsx3
import openai
import subprocess


"""
Get Voice over paragraph
"""
openai.api_key = "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"


def rewrite_paragraph(sentence):
    prompt = f"{sentence} \n\nRewrite the above sentence into a larger paragraph:\n"
    response = openai.Completion.create(
        engine="text-davinci-003",  # Specify the GPT-3 model
        prompt=prompt,
        max_tokens=100,  # Adjust the length of the generated text
        n=1,  # Number of responses to generate
        temperature=0.9,  # Adjust the randomness of the output
    )
    rewritten_paragraph = response.choices[0].text.strip()
    return rewritten_paragraph


"""
Generate the Voice over in an audio format

"""


def generate_voiceover(sentence, id, output_file, voice='female', speed_factor=1.0):
    # sentence = "Our product offers cutting-edge technology for small businesses."
    voice = 'male'  # Choose 'male' or 'female'
    engine = pyttsx3.init()

    # Get available voices
    voices = engine.getProperty('voices')

    # Set the desired voice based on the input parameter
    if voice == 'male':
        # Set the first available male voice
        engine.setProperty('voice', voices[0].id)
    elif voice == 'female':
        # Set the first available female voice
        engine.setProperty('voice', voices[1].id)

     # Generate speech
    temp_audio_file = output_file
    engine.save_to_file(sentence, temp_audio_file)
    engine.runAndWait()

    output_file = temp_audio_file
    if speed_factor != 1.0:
        # Modify audio speed
        audio = AudioSegment.from_file(temp_audio_file, format='wav')
        modified_audio = audio.speedup(playback_speed=speed_factor)

        # Save the modified audio
        output_file = 'voiceover.mp3'
        modified_audio.export(output_file, format='mp3')

        # Clean up temporary file
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)
    return output_file
# # Example usage

# sentence = "Our product offers cutting-edge technology for small businesses."
# voice = 'male'  # Choose 'male' or 'female'

# generate_voiceover(rewritten_paragraph, voice=voice, speed_factor=1.0)


"""
Add Background to the voice over

"""


def merge_audio_files(voiceover_file, background_file, output_file, voiceover_volume=1.0, background_volume=1.0):
    command = [
        'ffmpeg',
        '-i', voiceover_file,
        '-i', background_file,
        '-filter_complex', f'[0:a]volume={voiceover_volume}[v];[1:a]volume={background_volume}[b];[v][b]amix=inputs=2:duration=first:dropout_transition=2',
        output_file
    ]
    subprocess.run(command)
    return output_file


# Usage example
# merge_audio_files('temp.mp3', 'background.mp3', 'merged_audio.wav',voiceover_volume=1, background_volume=0.1)

"""
Add the voice over to the video

"""


def replace_audio(video_file, audio_file, output_file):
    command = [
        'ffmpeg',
        '-i', video_file,
        '-i', audio_file,
        '-c:v', 'copy',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        '-y',
        output_file
    ]
    subprocess.run(command)

    return output_file

# Usage example
# replace_audio('Test.mp4', 'temp.mp3', 'output_video.mp4')


"""
Execte the whole process

"""


def add_voice_over(input_video, choice, id):
    input_video = "media/"+str(input_video)
    # Make sure that Output folder exists
    output_folder = os.path.join(settings.MEDIA_ROOT, f'voiceOver/{id}')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    background = f'media/voiceOver/Background{choice}.mp3'
    # Replace Existing video audio with background
    final = replace_audio(
        input_video, background, output_file=f'media/videosOut/{id}/final_video_{id}.mp4')

    return f'videosOut/{id}/final_video_{id}.mp4'


# def add_voice_over(input_video, statement, id):
#     input_video = "media/"+str(input_video)
#     # Make sure that Output folder exists
#     output_folder = os.path.join(settings.MEDIA_ROOT, f'voiceOver/{id}')
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # 1 get Paragraph
#     paragraph = rewrite_paragraph(statement)

#     # 2 Generate voice over
#     voice_over = generate_voiceover(
#         paragraph, id, f'media/voiceOver/{id}/voice_{id}.mp3')

#     # 3 Merge with background
#     merged_with_background = merge_audio_files(
#         voice_over, 'media/voiceOver/Background.mp3', output_file=f'media/voiceOver/{id}/voice_background_{id}.mp3', voiceover_volume=1, background_volume=0.1)

#     # 3 Replace Existing video audi with voiceovers
#     final = replace_audio(
#         input_video, merged_with_background, output_file=f'media/videosOut/{id}/final_video_{id}.mp4')

#     return f'videosOut/{id}/final_video_{id}.mp4'
