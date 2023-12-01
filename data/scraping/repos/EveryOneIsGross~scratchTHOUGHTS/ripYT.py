'''
Run script.
Type project name when asked.
If project name is new, type YouTube video link.

Program downloads video.
Program saves transcript in two formats: readable and for video subtitles.
Program turns transcript into audio. Audio matches video's pacing. Is currently not same length : /

Ask questions about the video. Program answers using video's content.
To search transcript, type: 'search <your question>'.
To leave chat, type: 'exit'.
On exit, program saves chat history and searches as json.
'''

from pytube import YouTube
import json
import openai
import numpy as np
import pandas as pd
import os
from youtube_transcript_api import YouTubeTranscriptApi
from gpt4all import GPT4All, Embed4All
import time
from pydub import AudioSegment
import pyttsx3

insights = {
    "chat_interactions": [],
    "search_results": []
}

model = "mistral trismegistus"

OPENAI_ENGINE = "model"
OPENAI_API_KEY = 'null'
openai.api_key = OPENAI_API_KEY
openai.api_base = "http://localhost:4892/v1"

def download_video(video_url, output_folder):
    yt = YouTube(video_url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    video_duration = yt.length  # This gives the video duration in seconds
    stream.download(output_path=output_folder)
    return video_duration  # Return the duration for further processing

def extract_transcript(video_url, output_folder):
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_id = video_url.split("v=")[1].split("&")[0]  # Extract video ID from URL
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        data = []
        all_content = []  # List to store all the content
        for entry in transcript:
            data_entry = {
                "start": entry["start"],
                "end": entry["start"] + entry["duration"],
                "content": entry["text"]
            }
            data.append(data_entry)
            all_content.append(entry["text"])  # Append content to the list

        with open(f"{output_folder}/transcript.json", "w") as file:
            json.dump(data, file, indent=4)

        # Write only the content to a separate document
        with open(f"{output_folder}/content_only.txt", "w") as file:
            file.write(' '.join(all_content))  # Join content with a space and write to file

    except Exception as e:
        print(f"Error fetching transcript: {e}")


def embed_transcript(transcript_path, output_folder):
    with open(transcript_path, 'r') as file:
        transcript_data = json.load(file)
    
    embedder = Embed4All()  # Initialize the Embed4All model
    sentences = [entry["content"] for entry in transcript_data]
    embeddings = []

    for sentence in sentences:
        embedding = embedder.embed(sentence.strip())
        embeddings.append(embedding)

    df = pd.DataFrame({
        "text": sentences,
        "embedding": embeddings
    })
    df.to_csv(f"{output_folder}/word_embeddings.csv", index=False)


def distances_from_embeddings(query_embedding, embeddings, distance_metric='cosine'):
    if distance_metric == 'cosine':
        # Convert series to list of arrays and then stack them to form a 2D array
        embeddings = np.vstack(embeddings.tolist())
        
        # Normalize both query and embeddings for cosine similarity
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        distances = -np.dot(embeddings, query_embedding)  # Using negative because we will sort ascending
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")
    return distances

def get_embedding_text(api_key, prompt, embeddings_path):  
    embedder = Embed4All()  # Initialize the Embed4All model
    q_embedding = embedder.embed(prompt.strip())
    df = pd.read_csv(embeddings_path, index_col=0)
    df['embedding'] = df['embedding'].apply(eval).apply(np.array)

    df['distances'] = distances_from_embeddings(q_embedding, df['embedding'].values, distance_metric='cosine')
    returns = []
    for i, row in df.sort_values('distances', ascending=True).head(25).iterrows():
        returns.append(row.name)

    #return "\n".join([f"{i+1}. {segment} " for i, segment in enumerate(returns)])
    return " ... ".join(returns)  # Using "/" to separate segments

def chunk_content(file_path, chunk_size=64):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Split the content into words and chunk them
    words = content.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    
    return chunks

def convert_to_srt_old(transcript_path, output_folder):
    with open(transcript_path, 'r') as file:
        transcript_data = json.load(file)

    srt_content = ""
    for idx, entry in enumerate(transcript_data, start=1):
        start_time = seconds_to_srt_time(entry["start"])
        end_time = seconds_to_srt_time(entry["end"])
        text = entry["content"].replace("\n", " ")  # Ensure no line breaks in subtitle text

        srt_content += f"{idx}\n{start_time} --> {end_time}\n{text}\n\n"

    with open(f"{output_folder}/transcript.srt", "w") as srt_file:
        srt_file.write(srt_content)


def convert_to_srt(transcript_path, output_folder):
    with open(transcript_path, 'r') as file:
        transcript_data = json.load(file)

    srt_content = ""
    for idx, entry in enumerate(transcript_data, start=1):
        start_time = seconds_to_srt_time(entry["start"])
        
        # Check if this is the last subtitle entry or not
        if idx < len(transcript_data):
            # Make sure the end time doesn't overlap with the next subtitle's start time
            next_entry_start_time = transcript_data[idx]["start"]
            end_time = seconds_to_srt_time(min(entry["end"], next_entry_start_time - 0.1))
        else:
            end_time = seconds_to_srt_time(entry["end"])
        
        text = entry["content"].replace("\n", " ")  # Ensure no line breaks in subtitle text
        srt_content += f"{idx}\n{start_time} --> {end_time}\n{text}\n\n"

    with open(f"{output_folder}/transcript.srt", "w") as srt_file:
        srt_file.write(srt_content)


def seconds_to_srt_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = (seconds % 1) * 1000
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(milliseconds):03}"

def hhmmss_to_seconds(timestamp):
    # Check if the timestamp has the expected length
    if len(timestamp) != 6:
        raise ValueError("Timestamp should be in the format hhmmss.")
    
    # Check if all characters in the timestamp are digits
    if not timestamp.isdigit():
        raise ValueError("Timestamp should only contain numerical values.")
    
    hours, minutes, seconds = map(int, [timestamp[:2], timestamp[2:4], timestamp[4:]])
    return hours * 3600 + minutes * 60 + seconds

def summarize_text(api_key, text):
    def generate_response(api_key, prompt):
        one_shot_prompt = f'''Provide a concise summary of the following: {prompt}'''

        print(f"Input Prompt for Summary Agent: {one_shot_prompt}")
        completions = openai.Completion.create(
            model=model,
            #model_path = "C:\AI_MODELS\mistral-7b-instruct-v0.1.Q4_0.gguf",
            prompt=one_shot_prompt,
            max_tokens=1024,
            n=1,
            temperature=0.5,
        )
        message = completions.choices[0].text
        return message

    return generate_response(api_key, text)

def search_transcript(api_key, query, content_path, embeddings_path, top_n=3):  
    embedder = Embed4All()  # Initialize the Embed4All model
    q_embedding = embedder.embed(query.strip())
    
    # Chunk the content
    chunks = chunk_content(content_path)
    chunk_embeddings = [embedder.embed(chunk) for chunk in chunks]

    df = pd.DataFrame({
        "text_chunk": chunks,
        "embedding": chunk_embeddings
    })

    df['distances'] = distances_from_embeddings(q_embedding, df['embedding'].values, distance_metric='cosine')
    closest_segments = df.sort_values('distances', ascending=True).head(top_n)['text_chunk'].tolist()

    return closest_segments

def transcript_to_audio(transcript_path, output_folder, total_duration_seconds):
    with open(transcript_path, 'r') as file:
        transcript_data = json.load(file)
    
    engine = pyttsx3.init()
    
    # TTS Configuration
    default_rate = engine.getProperty('rate')
    engine.setProperty('rate', default_rate - 75)
    volume = engine.getProperty('volume')
    engine.setProperty('volume', 0.8)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)

    all_audio_segments = []

    # First, generate all TTS segments without adding silences
    for entry in transcript_data:
        text = entry["content"]

        # Calculate spoken duration at default rate
        words_per_minute = default_rate / 60
        words_in_text = len(text.split())
        spoken_duration = words_in_text / words_per_minute

        # Calculate actual duration from timecodes
        actual_duration = entry["end"] - entry["start"]

        # Adjust rate based on the ratio of spoken to actual durations
        adjusted_rate = default_rate * (spoken_duration / actual_duration)

        # Set bounds to ensure the adjusted rate doesn't go beyond certain limits
        min_rate = 0.75 * default_rate
        max_rate = 1 * default_rate
        adjusted_rate = max(min(adjusted_rate, max_rate), min_rate)

        engine.setProperty('rate', adjusted_rate)

        temp_file = f"{output_folder}/temp_{entry['start']}.wav"
        engine.save_to_file(text, temp_file)
        engine.runAndWait()
        all_audio_segments.append(AudioSegment.from_wav(temp_file))

        os.remove(temp_file)

    # Now, intersperse with calculated silences
    final_audio_list = []
    for index, audio_segment in enumerate(all_audio_segments):
        # Add the TTS audio segment
        final_audio_list.append(audio_segment)

        # If this isn't the last segment, calculate and add the silence
        if index < len(all_audio_segments) - 1:
            next_entry = transcript_data[index + 1]
            silence_duration = (next_entry["start"] - transcript_data[index]["end"]) * 1000
            silence = AudioSegment.silent(duration=silence_duration)
            final_audio_list.append(silence)

    # Concatenate all segments to produce the final audio
    final_audio = sum(final_audio_list)
    
    final_audio_file = f"{output_folder}/transcript_audio.mp3"
    final_audio.export(final_audio_file, format="mp3")

    print(f"Transcript audio saved to {final_audio_file}")




def chat_with_transcript(api_key, user_input, embeddings_path):
    def generate_response(api_key, prompt):
        one_shot_prompt = f'''Based on the given context, answer the question: {prompt}'''

        print(f"Input Prompt for Agent: {one_shot_prompt}")
        completions = openai.Completion.create(
            model=model,
            prompt=one_shot_prompt,
            max_tokens=1024,
            #n=1, # Number of responses to return
            temperature=0.4,
            #stop=["\n\n"]
        )
        message = completions.choices[0].text
        return message

    text_embedding = get_embedding_text(api_key, user_input, embeddings_path)  # Use the main function
    user_input_embedding = f'Using this context: "{text_embedding}", answer the following question: \n{user_input}'

    response = generate_response(api_key, user_input_embedding)
    #print(f"Response from Agent: {response}")
    # Update insights
    insights["chat_interactions"].append({
        "question": user_input,
        "response": response.strip()
    })
    
    return response.strip()

if __name__ == '__main__':
    project_name = input("Enter the project name: ")
    output_folder = os.path.join(os.getcwd(), project_name)
    embeddings_path = f"{output_folder}/word_embeddings.csv"  # Define this here for consistency
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created new project folder named '{project_name}'.")
        video_url = input("Enter the YouTube video URL: ")

        # Extract the transcript and convert it to .srt format
        extract_transcript(video_url, output_folder)
        transcript_path = f"{output_folder}/transcript.json"
        convert_to_srt(transcript_path, output_folder)

        # Embed the transcript
        embed_transcript(transcript_path, output_folder)
        
        # Download the video and get its duration
        video_duration = download_video(video_url, output_folder)
        
        # Convert the transcript to audio
        transcript_to_audio(transcript_path, output_folder, video_duration)
    
    else:
        print(f"Found existing project folder named '{project_name}'.")
        if not os.path.exists(embeddings_path):
            print("No embeddings found in the project folder.")
            video_url = input("Enter the YouTube video URL: ")

            # Extract the transcript and convert it to .srt format
            extract_transcript(video_url, output_folder)
            transcript_path = f"{output_folder}/transcript.json"
            convert_to_srt(transcript_path, output_folder)

            # Embed the transcript
            embed_transcript(transcript_path, output_folder)
            # Convert the transcript to audio
            transcript_to_audio(transcript_path, output_folder)
            download_video(video_url, output_folder)
        else:
            print("Embeddings already exist in the specified folder.")


while True:
    question = input("Ask me something about the video, 'search <your query>' to search (or type 'exit' to quit): ")

    if question.lower() == 'exit':
        insights_filename = os.path.join(output_folder, f"{project_name}_insights.json")
        
        # Read the existing content
        if os.path.exists(insights_filename):
            with open(insights_filename, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = {"chat_interactions": [], "search_results": []}
        
        # Update the in-memory data structure
        existing_data['chat_interactions'].extend(insights['chat_interactions'])
        existing_data['search_results'].extend(insights['search_results'])

        # Write the data back to the file
        with open(insights_filename, 'w') as f:
            json.dump(existing_data, f, indent=4)

        break
    elif question.lower().startswith('search '):
        search_query = question.split('search ', 1)[1]
        results = search_transcript(OPENAI_API_KEY, search_query, f"{output_folder}/content_only.txt", embeddings_path)
        
        # Update insights
        insights["search_results"].append({
            "query": search_query,
            "results": results
        })
        
        print("Top search results from the transcript:")
        for idx, segment in enumerate(results, start=1):
            print(f"{idx}. {segment}")

    else:
        answer = chat_with_transcript(OPENAI_API_KEY, question, embeddings_path)
        print(f"Answer: {answer}")

