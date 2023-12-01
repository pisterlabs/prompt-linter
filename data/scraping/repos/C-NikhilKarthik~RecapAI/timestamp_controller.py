from youtube_transcript_api import YouTubeTranscriptApi
from flask import Blueprint, request, jsonify
import difflib
import openai
from dotenv import load_dotenv
import os
import re
from utilities.transcript import transcript

timestamp_controller = Blueprint("timestamp_controller", __name__)


openai_api_key = os.getenv("OPENAI_KEY")

# Initialize the OpenAI API client
openai.api_key = openai_api_key
@timestamp_controller.route('/timestamp', methods=['POST'])
def timestamp():
    final_data = {}

    data = request.get_json()
    video_url = data.get('videoURL')

    timestamped_data = YouTubeTranscriptApi.get_transcript(video_url)

    if video_url is None:
        return jsonify({"error": "Missing 'videoURL' parameter"}), 400

    # Check if the transcript is already cached, and if not, transcribe the video
    cached_transcript = transcript(video_url)

    user_message = cached_transcript + "Extract ten important sentences from the given transcript that would be valuable for a viewer and return them in a python list."

    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_message}],
    )

    response_text = chat_completion['choices'][0]['message']['content']

    points = re.findall(r'\d+\.\s+"(.*?)"', response_text)

    matches = {}

    # Iterate over sentences from the first array
    for sentence1 in points:
        best_match = None
        best_ratio = 0
        
        # Iterate over sentences from the second array
        for sentence2 in timestamped_data:
            # Calculate similarity ratio using difflib's SequenceMatcher
            similarity_ratio = difflib.SequenceMatcher(None, sentence1, sentence2["text"]).ratio()
            
            # Update the best match if a better match is found
            if similarity_ratio > best_ratio:
                best_ratio = similarity_ratio
                best_match = sentence2
        
        if best_match:
            # Convert start_time to an integer
            start_time = int(best_match["start"]) - 1
            matches[sentence1] = {"start": start_time, "duration": best_match["duration"]}

    # Now 'matches' contains matched sentences with start times
    for sentence, data in matches.items():
        final_data[sentence] = {
            "start_time": data["start"],
            "duration": data["duration"]
        }

    return jsonify({"timestamps": final_data}), 200