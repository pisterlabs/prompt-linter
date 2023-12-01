import youtube_dl
from youtube_transcript_api import YouTubeTranscriptApi
from googletrans import Translator
from pytube import YouTube
import time
from translate import Translator
import re
import json
import openai
import time

#todo
'''
the cadence is not so important.

translate sentence by sentence

reformat cadence
Give timestamp for every sentence if it is not one word long
translate by sentences (check the number of sentences)
write it back
'''


# Set up OpenAI API key
openai.api_key = "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"


# Set up OpenAI GPT-3 language translation model
model_engine = "text-davinci-002"
model_prompt = ("<|model:{}|>"
                "<|source-lang:en|><|target-lang:hu|>".format(model_engine))






def translate_text_file(input_filename, output_filename, chunk_size=2048):
    # Read the input file
    with open(input_filename, "r") as input_file:
        input_text = input_file.read()

    # Divide the input text into chunks based on the maximum token size
    input_chunks = []
    for i in range(0, len(input_text), chunk_size):
        input_chunks.append(input_text[i:i+chunk_size])

    # Translate each chunk using the OpenAI GPT-3 API
    translated_chunks = []
    for i, input_chunk in enumerate(input_chunks):
        # Add the model prompt to the input text


        # prompt = model_prompt + input_chunk
        # print(prompt)
        # Translate the input text using the OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content":
                    '''
                     Translate the following text to Hungarian,
                     the numbers serves as placholder for timestamps in an srt file.
                     The text is the folllowing:
                    ''' + input_chunk}
            ]
        )



        # response = openai.Completion.create(
        #     engine=model_engine,
        #     prompt=prompt,
        #     max_tokens=2048,
        #     n=1,
        #     stop=None,
        #     temperature=0.5,
        # )
        print(response['choices'][0]['message']['content'])
        # Get the translated text from the API response
        translated_text = response.choices[0].text[len(input_chunk):].strip()

        # Append the translated text to the list of translated chunks
        translated_chunks.append(translated_text)

        # Sleep for a short time to avoid hitting the API rate limit
        time.sleep(0.5)

    # Merge the translated chunks into a single string
    translated_text = "".join(translated_chunks)

    # Write the translated text to the output file
    with open(output_filename, "w") as output_file:
        output_file.write(translated_text)


def get_video_info(video_url):
    try:
        yt = YouTube(video_url)
        video_info = {
            'id': yt.video_id,
            'title': yt.title,
            'thumbnail_url': yt.thumbnail_url,
            'length': yt.length,
            'views': yt.views,
            'rating': yt.rating,
        }
        return video_info
    except Exception as e:
        print(f"Error getting video info: {e}")
        return None


def remove_timestamp_from_original(filename, map_filename):
    # Open the SRT file for reading
    with open(filename, 'r') as file:
        data = file.read()

    # Replace the timestamps with empty strings
    timestamp_pattern = r'(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)\n'
    timestamps = re.findall(timestamp_pattern, data)
    data = re.sub(timestamp_pattern, '', data)

    # Create a dictionary with ID as key and corresponding timestamp as value
    id_timestamp_map = {}
    for i, timestamp in enumerate(timestamps):
        id_timestamp_map[str(i + 1)] = timestamp

    # Save the map to a JSON file
    with open(map_filename, 'w') as file:
        json.dump(id_timestamp_map, file)

    # Save the modified data to a new file
    with open('new_' + filename, 'w') as file:
        file.write(data)


def add_timestamps_to_tranlated(filename, map_filename):
    # Load the ID-timestamp map from the JSON file
    with open(map_filename, 'r') as file:
        id_timestamp_map = json.load(file)

    # Open the SRT file for reading
    with open(filename, 'r') as file:
        data = file.readlines()

    # Create a new list to store the modified data
    new_data = []

    # Iterate through the lines in the original file
    for i in range(0, len(data), 2):
        # Get the ID and text from the original file
        id = data[i].strip()
        text = data[i + 1].strip()

        # Add the timestamp from the id_timestamp_map
        timestamp = id_timestamp_map[id]
        new_data.append(id)
        new_data.append(timestamp[0] + ' --> ' + timestamp[1])
        new_data.append(text)
        new_data.append('')

    # Save the modified data to a new file
    with open('new_' + filename, 'w') as file:
        file.writelines(new_data)


def export_subtitles_to_indexed_srt(subtitles, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for i, subtitle in enumerate(subtitles):
            f.write(f"{i + 1}\n")
            f.write(f"{subtitle['text']}\n")
            f.write("\n")


def get_subtitles(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(['en'])
        subtitles = transcript.fetch()
        filename = 'subtitles.srt'
        export_subtitles_to_srt(subtitles, filename)
        remove_timestamp_from_original(filename, 'map.json')

        translate_text_file('new_' + filename, 'hu_' + filename, chunk_size=248)
        import sys
        sys.exit()

        return subtitles
    except Exception as e:
        print(f"Error fetching subtitles: {e}")
        return None


def export_subtitles_to_srt(subtitles, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for i, subtitle in enumerate(subtitles):
            f.write(f"{i + 1}\n")
            start_time = format_timestamp(subtitle['start'])
            end_time = format_timestamp(subtitle['start'] + subtitle['duration'])
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{subtitle['text']}\n\n")


def format_timestamp(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def translate_subtitles(subtitles, target_language='hu'):
    translator = Translator(to_lang=target_language)
    translated_subtitles = []

    for i, subtitle in enumerate(subtitles):
        try:
            translated_text = translator.translate(subtitle['text'])
            translated_subtitles.append({
                'start': subtitle['start'],
                'duration': subtitle['duration'],
                'text': translated_text,
            })
            print(f"Translated subtitle {i + 1}/{len(subtitles)}")
        except Exception as e:
            print(f"Error translating subtitle {i + 1}/{len(subtitles)}: {e}")

    return translated_subtitles

if __name__ == "__main__":


    video_url = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
    video_url = 'https://www.youtube.com/watch?v=cab-58TyBd0'
    video_url = 'https://www.youtube.com/watch?v=tV1dN20e_vY'
    video_url = 'https://www.youtube.com/watch?v=pHCb3bpBxW0'
    video_info = get_video_info(video_url)

    if video_info is not None:
        video_id = video_info['id']
        print(video_id)
        subtitles = get_subtitles(video_id)
        print(subtitles)
        if subtitles:
            translated_subtitles = translate_subtitles(subtitles)
            print(translated_subtitles)
            for subtitle in translated_subtitles:
                print(f"{subtitle['start']:.2f} - {subtitle['text']}")
        else:
            print("No subtitles found.")
    else:
        print("Video information could not be retrieved.")
