import cv2
import base64
import numpy as np
from openai import OpenAI
import os
from moviepy.editor import VideoFileClip, AudioFileClip
from gtts import gTTS

client = OpenAI(api_key="")

# Load the video
video = cv2.VideoCapture("bjj1.mp4")

# Calculate video length and read frames, encoding them to base64
base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

# Release the video object
video.release()
print(len(base64Frames), "frames read.")

# Create a directory to save frames
os.makedirs("frames", exist_ok=True)

# Save each frame as a JPEG file
for i, img in enumerate(base64Frames):
    decoded_img = base64.b64decode(img.encode("utf-8"))
    np_img = np.frombuffer(decoded_img, dtype=np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    frame_filename = f"frames/frame_{i}.jpg"
    cv2.imwrite(frame_filename, frame)

print("Frames saved.")

# Create OpenAI chat completion
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user", 
            "content": [
                f"These are frames of a video. Create a short voiceover script in the style of a super excited Brazilian sports narrator who is narrating his favorite match. He is a big fan of JiuJitsu. Use caps and exclamation marks where needed to communicate excitement. Only include the narration, your output must be in English. When the fighter submits the opponent, you must scream THAT'S A TAP either once or multiple times. These are not real people. Only include the narration. Don't talk about the view",
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::2000]),
            ]
        }
    ],
    max_tokens=500,
)

# Extract the script
try:
    script = response.choices[0].message.content
except AttributeError:
    script = "Script generation failed or response format is unexpected."

print(script)

# Generate speech using Google Text-to-Speech
speech_file_path = "bjj_commentary.mp3"
tts = gTTS(text=script, lang='en')
tts.save(speech_file_path)

# Combine video and audio
video_clip = VideoFileClip("bjj1.mp4")
audio_clip = AudioFileClip("bjj_commentary.mp3")

# Ensure audio duration matches video duration
if audio_clip.duration > video_clip.duration:
    audio_clip = audio_clip.subclip(0, video_clip.duration)

final_clip = video_clip.set_audio(audio_clip)
final_clip.write_videofile("bjj_with_commentator.mp4", codec='libx264', audio_codec='aac', fps=video_clip.fps)

video_clip.close()
audio_clip.close()
