import openai
import json
import time
from concurrent.futures import ThreadPoolExecutor

with open('transcription.json', 'r') as file:
    transcription_data = json.load(file)

# Set OpenAI API key here
openai.api_key = 'openai-api-key'

highlights = []

def get_highlight(segment):
    prompt = f"Is the following segment from a basketball game commentary a highlight? If yes, summarize it.\n\nSegment: \"{segment['text']}\""

    try:
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=30,
            temperature=0.3
        )
        
        content = response.choices[0].text.strip()
        if content.lower().startswith("yes"):
            summary = content[len("yes"):].strip().lstrip(',').strip()
            return {
                'segment_id': segment['id'],
                'start_time': segment['start'],
                'end_time': segment['end'],
                'highlight': summary
            }
    except openai.error.OpenAIError as e:
        print(f"An error occurred with segment ID {segment['id']}: {e}")
    except openai.error.RateLimitError:
        print(f"Rate limited on segment ID {segment['id']}. Retrying...")
        time.sleep(0.001)
        return get_highlight(segment)

def handle_future(future):
    result = future.result()
    if result:
        highlights.append(result)

executor = ThreadPoolExecutor(max_workers=5)

futures = [executor.submit(get_highlight, segment) for segment in transcription_data['segments']]

for future in futures:
    future.add_done_callback(handle_future)

executor.shutdown(wait=True)

with open('highlights.json', 'w') as file:
    json.dump(highlights, file, indent=4)
