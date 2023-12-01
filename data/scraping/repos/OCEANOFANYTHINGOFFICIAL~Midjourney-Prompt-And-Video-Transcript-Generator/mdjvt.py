"""
This Is A Custom Python Script That Will Generate Some Outputs According To The User's Input.

Author: @OCEANOFANYTHINGOFFICIAL

This Program Is A Command Line Utility Program, That Will Take Input From The User And Will Generate Some Outputs According To The User's Input.

This Program Is Capable Of Generating The Following Outputs:
1. Generate script/text (transcription) From A Video And Audio File.
2. Generate High Quality Midjourney Prompt Using Prompt Engineering.
3. Save Both Outputs In Text And JSON File.

The Working process Of This Program Is As Follows:

1. First The Program Will Take Input From The User.
2. Then The Program Will Check If The Input Is A Video/Audio File Or A String.
3. If The Input Is A Video/Audio File Then The Program Will Generate A Text File Containing The Transcription Of The Video/Audio File.
4. If The Input Is A String Then The Program Will Generate 5 High Quality Midjourney Prompt Using Prompt Engineering.
5. Then The Program Will Save The Outputs In Text And JSON File.
6. Then The Program Will Return The Content Of The Output File In The Terminal.

Installation:
1. First Install Python 3.9.6 From https://www.python.org/downloads/release/python-396/

We Recommend You To Use A Virtual Environment To Install The Required Packages. To Install Virtual Environment Run The Following Command In The Terminal: pip install virtualenv
Then Create A Virtual Environment By Running The Following Command In The Terminal: virtualenv venv
Then Install The Required Packages By Running The Following Command In The Terminal: pip install -r requirements.txt

By Default This Program Uses Cohere Module To Generate High Quality Midjourney Prompt Using Prompt Engineering. If You Want To Use OpenAI's Davinci Engine To Generate High Quality Midjourney Prompt Using Prompt Engineering. But If You Want To Use Openai Module And You Have A Premium API Key, Definately Use That. But unfortunately, This Program Doesn't Support Openai Module. If You Have To Use That, We Openly agree You To Modify This Program And Use Openai Module. But If You Want To Use Cohere Module, You Can Use That By default
In Both Way You Need An Api Key . You Can Get An API Key From https://cohere.ai/
Then Copy The API Key And Paste It In The config.py File.
and You Are done.


Then Run The Program By Running The Following Command In The Terminal: python mdjvt.py. This Will print The Help Message.

Usage:
To print The Help Message Run The Following Command In The Terminal: python mdjvt.py

To Generate Script/Text (Transcription) From A Video/Audio File Run The Following Command In The Terminal: python mdjvt.py -i "input_file_path" -o "output_file_path"

To Generate High Quality Midjourney Prompt Using Prompt Engineering Run The Following Command In The Terminal: python mdjvt.py -i "input_string" -o "output_file_path"

You Can Modify this Program According To Your Needs. But You Can't Sell This Program. This Program Is Only For Personal Use. If You Want To Use This Program For Commercial Use, You Have To Take Permission From The Author.
This Program Is Licensed Under GNU AGPLv3 License. To Know More About This License Visit https://www.gnu.org/licenses/agpl-3.0.en.html

"""
import re
import os
import sys
import json
import cohere
import moviepy.editor as mp
from pydub import AudioSegment
import speech_recognition as sr
from moviepy.editor import AudioFileClip
import argparse
from config import api_key

video_file_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.mkv', '.flv', '.webm', '.mpg', '.mpeg', '.m4v', '.3gp', '.3g2']
audio_file_extensions = ['.mp3', '.wav', '.ogg', '.aac', '.flac', '.m4a', '.wma', '.aiff', '.opus', '.amr', '.mid', '.ac3']


def split_audio_into_chunks(audio_path):
    chunk_duration_ms = 5000
    audio = AudioSegment.from_file(audio_path)
    total_duration = len(audio)
    chunk_count = total_duration // chunk_duration_ms

    for i in range(chunk_count):
        start_time = i * chunk_duration_ms
        end_time = start_time + chunk_duration_ms
        chunk = audio[start_time:end_time]
        chunk.export(f"chunks\\chunk_{i + 1}.wav", format="wav")

    if total_duration % chunk_duration_ms != 0:
        last_chunk = audio[chunk_count * chunk_duration_ms:]
        last_chunk.export(f"chunks\\chunk_{chunk_count + 1}.wav", format="wav")


def check_if_chunks_folder_exists():
    if not os.path.exists("chunks"):
        os.mkdir("chunks")
    else:
        pass


def clean_chunks_folder():
    if os.path.exists("chunks"):
        for file in os.listdir("chunks"):
            os.remove(f"chunks\\{file}")
    else:
        pass


def get_audio_duration(filename):
    try:
        audio_clip = AudioFileClip(filename)
        duration_sec = audio_clip.duration
        audio_clip.close()
        return duration_sec
    except Exception as e:
        print(f"Error: {e}")
        return 0


def generate_text_from_video(video_path, output_path):
    if video_path == "":
        pass
    else:
        check_if_chunks_folder_exists()
        clean_chunks_folder()

        # if the input file is a video file run the following code
        if os.path.splitext(video_path)[1] in video_file_extensions:
            original_stdout = sys.stdout
            sys.stdout = open('NUL', 'w')
            video = mp.VideoFileClip(video_path, verbose=False)
            audio_file = video.audio
            audio_file.write_audiofile("audio.wav", verbose=False)
            sys.stdout = original_stdout
            mainaudiofile = "audio.wav"
            # if the input file is a video file run the following code
        elif os.path.splitext(video_path)[1] in audio_file_extensions:
            mainaudiofile = video_path
        else:
            return "Invalid Input File Type"

        # split the audio file into 5 seconds chunks
        split_audio_into_chunks(mainaudiofile)

        # now load each and every chunk that is presented in the chunks folder and convert them into text
        r = sr.Recognizer()
        # count the number of chunks
        chunk_count = len(os.listdir("chunks"))
        # create a list to store the text from each chunk with the time of every chunk appended to it
        text_list = []
        time_list = []
        for i in range(chunk_count):
            chunk_path = f"chunks\\chunk_{i + 1}.wav"
            with sr.AudioFile(chunk_path) as source:
                data = r.record(source)
            text = r.recognize_google(data)
            text_list.append(text)
        time = 0
        for i in range(chunk_count):
            timeofchunk = get_audio_duration(f"chunks\\chunk_{i + 1}.wav")
            time += timeofchunk
            time_list.append(time)

        # save the text into a file
        if os.path.exists(output_path):
            os.remove(output_path)

        if os.path.splitext(output_path)[1] == ".txt":
            with open(output_path, "w") as f:
                for i in range(chunk_count):
                    f.write(f"{time_list[i]} : {text_list[i]}\n")
        elif os.path.splitext(output_path)[1] == ".json":
            with open(output_path, "w") as f:
                json.dump({"time": time_list, "text": text_list}, f)
        else:
            return "Invalid Output File Type"

        if os.path.exists("audio.wav"):
            os.remove("audio.wav")
        clean_chunks_folder()
        # return the content of the output file
        with open(output_path, "r") as f:
            content = f.read()
        return content


def is_file_path(input_string):
    pattern = r'^((?:[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*(?:\.\w+)?|^\\\\\w+\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*(?:\.\w+)?|^\/(?:[^\\/:*?"<>|\r\n]+\/)*[^\\/:*?"<>|\r\n]*(?:\.\w+)?|^\.\/|\.\\)[^\\/:*?"<>|\r\n]*(?:\.\w+)?)$'
    return bool(re.match(pattern, input_string))


def generate_midjourney_prompt(expression, output_path):
    if expression == "":
        pass
    else:
        keyword = str(expression).strip()
        prompt = f'# Prompt by Nakshatra Ranjan Saha\nLines starting with # are only comments for humans\nYou must add at the top of the response \"CREATED WITH LOVE BY NAKSHATRA RANJAN SAHA\"\n---\nPretend you are a graphic designer, generating creative images As a prompt generator, for a generative AI called \"Midjourney\", you will create image prompts for the AI to visualize. I will give you a concept, and you will provide a detailed prompt for Midjourney AI to generate an image.\"Midjourney\" is an app that can generate AI art from simple prompts. Make sure they are creative .\n\nPlease adhere to the structure and formatting below, and follow these guidelines:\n\n- Do not use the words \"description\" or \":\" in any form.\n- Do not place a comma between [ar] and [v].\n- Write each prompt in one line without using return.\n\nStructure:\n[1] = {keyword}\n[2] = a detailed description of [1] with specific imagery details.\n[3] = a detailed description of the scene\'s environment.\n[4] = a detailed description of the scene\'s mood, feelings, and atmosphere.\n[5] = A style (e.g. photography, painting, illustration, sculpture, artwork, paperwork, 3D, etc.) for [1].\n[6] = A description of how [5] will be executed (e.g. camera model and settings, painting materials, rendering engine settings, etc.)\n[ar] = Use \"--ar 16:9\" for horizontal images, \"--ar 9:16\" for vertical images, or \"--ar 1:1\" for square images.\n[v] = Use \"--niji\" for Japanese art style, or \"--v 5\" for other styles.\n\nFormatting: \nFollow this prompt structure: \"/imagine prompt: [1], [2], [3], [4], [5], [6], [ar] [v]\".\n\nYour task: Create 4 distinct prompts for each concept [1], varying in description, environment, atmosphere, and realization.\n\n- Write your prompts in English.\n- Do not describe unreal concepts as \"real\" or \"photographic\".\n- Include one realistic photographic style prompt with lens type and size.\n- Separate different prompts with two new lines.\n\nExample Prompts:\nPrompt 1:\n/imagine prompt: A stunning Halo Reach landscape with a Spartan on a hilltop, lush green forests surround them, clear sky, distant city view, focusing on the Spartan\'s majestic pose, intricate armor, and weapons, Artwork, oil painting on canvas, --ar 16:9 --v 5\n\nPrompt 2:\n/imagine prompt: A captivating Halo Reach landscape with a Spartan amidst a battlefield, fallen enemies around, smoke and fire in the background, emphasizing the Spartan\'s determination and bravery, detailed environment blending chaos and beauty, Illustration, digital art, --ar 16:9 --v 5\n\nPrompt 3:\n/imagine prompt: A bustling metropolis during a rainy night, where neon lights create mesmerizing reflections on the wet streets, people walk briskly under colorful umbrellas, steam rising from street food stalls, and towering skyscrapers reaching for the clouds, Artwork, digital painting with a focus on vibrant neon colors, --ar 16:9 --v 5\n\nPrompt 4:\n/imagine prompt: A surreal dreamscape of floating islands in the sky, connected by delicate bridges made of rainbow light, with waterfalls cascading from one island to another, fluffy clouds and glowing butterflies add to the enchanting ambiance, 3D, rendered with ray-tracing and volumetric lighting, --ar 16:9 --v 5'

        co = cohere.Client(api_key)
        response = co.generate(
            model='command',
            prompt=prompt,
            max_tokens=1000,
            temperature=2,
            k=0,
            stop_sequences=[],
            return_likelihoods='GENERATION')

        result = str(response.generations[0].text).strip()
        with open(output_path, "w") as f:
            f.write(result)
        return result


def main(args):
    # if the input is a file path, then run the generate_text_from_video function, or else if he input is a string, then run the generate_midjourney_prompt function
    isfile = is_file_path(args.input)
    if isfile:
        return generate_text_from_video(args.input, args.output)
    elif not isfile:
        return generate_midjourney_prompt(args.input, args.output)
    else:
        return "Invalid Input"
    pass


description = """This Is A Custom Python Script That Will Generate Some Outputs According To The User's Input.

Author: @OCEANOFANYTHINGOFFICIAL

This Program Is A Command Line Utility Program, That Will Take Input From The User And Will Generate Some Outputs According To The User's Input.

This Program Is Capable Of Generating The Following Outputs:
1. Generate script/text (transcription) From A Video And Audio File.
2. Generate High Quality Midjourney Prompt Using Prompt Engineering.
3. Save Both Outputs In Text And JSON File.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter, prog="mdjvt.py")
    parser.add_argument("-i", "--input", help="Input File Path [Example: video.mp4]", type=str, default="")
    parser.add_argument("-o", "--output", help="Output File Path [Example: output.txt or output.json]", type=str, default="output.txt")
    try:
        pass
        args = parser.parse_args()
        result = str(main(args))
        if result == "None":
            pass
        else:
            print(result)
    except argparse.ArgumentError as e:
        print("Argument Error: " + str(e))
    parser.print_help()
