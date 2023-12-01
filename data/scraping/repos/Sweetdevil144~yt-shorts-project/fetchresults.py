import openai
from api_keys import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY


def extract_shorts(captions):
    chunks = divide_captions_into_chunks(captions, min_duration=15, target_duration=30,
                                         max_duration=35)  # Set min, target, and max duration for each chunk
    ratings = []

    for chunk in chunks:
        text = ' '.join([caption['text'] for caption in chunk])
        rating = analyze_captions(text)
        start_time = chunk[0]['start']
        end_time = chunk[-1]['start'] + chunk[-1]['duration']
        ratings.append((rating, start_time, end_time))

    # Select the top-rated chunks
    top_shorts = sorted(ratings, reverse=True)[:3]  # You can change this to 4 if you need four shorts
    timestamps = [(start_time, end_time) for _, start_time, end_time in top_shorts]
    return timestamps


def divide_captions_into_chunks(captions, min_duration, target_duration, max_duration):
    chunks = []
    current_chunk = []
    current_time = 0
    max_tokens = 16200  # Updated for gpt-3.5-turbo-16k

    for caption in captions:
        caption_tokens = len(caption['text'].split())
        next_time = current_time + caption['duration']

        if (next_time >= target_duration and next_time - current_time >= min_duration) or next_time > max_duration:
            chunks.append(current_chunk)
            current_chunk = [caption]
            current_time = caption['duration']
        else:
            current_chunk.append(caption)
            current_time += caption['duration']

    if current_chunk and current_time >= min_duration:
        chunks.append(current_chunk)

    return chunks


def analyze_captions(text):
    conversation = [
        {"role": "system",
         "content": "You are a helpful assistant that analyzes video transcripts to identify the best parts for creating YouTube shorts."},
        {"role": "user",
         "content": f"I want to create YouTube shorts from a single video. Analyze the following text and tell me if it would make a good short: {text}"},
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=conversation
    )
    # Extract rating from response. You will need to define logic here to extract a rating from the response message
    rating = len(response['choices'][0]['message']['content'])
    return rating


def generate_youtube_link(video_id, start_time, end_time):
    base_url = f"https://www.youtube.com/watch?v={video_id}"
    link_with_time = f"{base_url}&t={round(start_time)}s"
    return link_with_time
