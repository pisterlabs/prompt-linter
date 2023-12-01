import re
import cv2
import base64
import openai
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip

def convert_timestamp_to_seconds(timestamp):
    minutes, seconds = map(int, timestamp.split(':'))
    return minutes * 60 + seconds

# Initialize OpenAI client
client = openai.OpenAI()

### VIDEO TO TEXT ###
video = cv2.VideoCapture("football.mp4")


# Calculate video length
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)
video_length_seconds = length / fps

print(f'Video length: {video_length_seconds:.2f} seconds')


# Get the frame rate of the video
fps = video.get(cv2.CAP_PROP_FPS)
# Calculate the number of frames for 11 seconds
frames_for_11_seconds = int(fps * 11)

# Read frames and encode to base64
base64Frames = []
frame_count = 0

while video.isOpened():
    success, frame = video.read()
    if not success:
        break

    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    frame_count += 1
# Release video object
video.release()
print(len(base64Frames), "frames read.")


# Prompt
prompt = f"These are selected frames from a {video_length_seconds:.2f} long football match video. Please create a highly concise scripted voiceover for a football commentary, segmented by timestamps. Each comment should be brief, ideally fitting within a 5-second window, ensuring the total commentary does not exceed {video_length_seconds:.2f} seconds. Format: [00:00] 'Quick observation...' [00:05] 'Another brief comment...' Tailor the comments to match the style of a succinct football game commentary."

# Set a higher temperature for creativity, if desired
temperature = 1

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[{
        "role": "user",
        "content": [
                prompt,
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::222]),
            ]
    }],
    max_tokens=500
)
script = response.choices[0].message.content
print(script)

# Regular expression to find all timestamped segments
# this regular expression matches segments of text that start with a timestamp in square brackets, 
# followed by commentary text in quotes, and ensures that each segment is followed by either another 
# timestamp or the end of the string
pattern = re.compile(r"\[\s*(\d{2}:\d{2})\s*\]\s*\"(.*?)\"(?=\s*\[\d{2}:\d{2}\]|$)", re.DOTALL)

# Find all matches
segments = pattern.findall(script)

# Check the structure of segments
print("Number of segments found:", len(segments))
for segment in segments:
    print(segment)

## Text to Speech ##
for i, (timestamp, text) in enumerate(segments):
    # Remove or adjust the timestamp in the text, if necessary
    segment_text = text  # Adjust as needed

    # Call the TTS API for each segment
    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=segment_text
    )
    
    # Save each audio clip
    clip_path = f"segment_{i}.mp3"
    print(clip_path)
    response.stream_to_file(clip_path)


## Merge video and audio ##

# Load the video clip
video_clip = VideoFileClip("football.mp4")

# Initialize an array to hold all audio clips with their start times
audio_clips_with_start_times = []
end_time_of_last_clip = 0

# Create the audio clips and put them into the array
for i, (timestamp, _) in enumerate(segments):
    start_time = convert_timestamp_to_seconds(timestamp)
    audio_clip = AudioFileClip(f"segment_{i}.mp3").set_start(start_time)
    
    # Ensure the new clip doesn't start before the previous one ends
    start_time = max(start_time, end_time_of_last_clip)
    
    # Set the start time and add the clip to the list
    audio_clips_with_start_times.append(audio_clip.set_start(start_time))

    # Update the end time of the last clip
    end_time_of_last_clip = start_time + audio_clip.duration


# Combine all audio clips
final_audio = CompositeAudioClip(audio_clips_with_start_times)

# Set the combined audio to the video clip
final_clip = video_clip.set_audio(final_audio)

# Export the final video
final_clip.write_videofile("football_with_segmented_commentary.mp4", codec="libx264", audio_codec="aac")

# Close the clips
video_clip.close()
final_clip.close()
for audio_clip in audio_clips_with_start_times:
    audio_clip.close()