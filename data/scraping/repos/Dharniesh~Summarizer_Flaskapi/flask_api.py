import openai
import whisper
import subprocess
import tiktoken
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
# from config import api_key
app = Flask(__name__)

openai.api_key=openai_api
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")

def chunker(text, max_tokens=200):
    token_count = len(encoding.encode(text))
    return (token_count + max_tokens - 1) // max_tokens

def split_string(string, n):
    part_length = len(string) // n
    return [string[i:i + part_length] for i in range(0, len(string), part_length)]

def summarize_aud(chun_txt):
    prompt1 = """
    You are a helpful assistant that summarizes videos.
    You are provided chunks of raw audio that were transcribed from the video's audio.
    Summarize the current chunk to succinct and clear bullet points of its contents.
    """

    prompt = prompt1 + chun_txt
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=700,
        temperature=0.7,
    )
    return response.choices[0].message.content

def mp42wav(input_file, output_file="audio_file.mp3"):
    command = [
        "ffmpeg", "-i", input_file, "-y", "-vn", "-acodec", "libmp3lame", "-ar", "44100", "-ac", "2", output_file
    ]
    subprocess.call(command)
    return output_file

def wav2txt(output_file):
    model = whisper.load_model("base")
    result = model.transcribe(output_file, language='en')
    with open("transcription.txt", "w", encoding="utf-8") as txt:
        txt.write(result["text"])
    with open('transcription.txt', 'r') as file:
        trans = file.read()
    return trans

@app.route('/upload_video', methods=['POST','GET'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    video_file = request.files['file']

    if video_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if video_file:
        # Securely save the uploaded video
        video_filename = secure_filename(video_file.filename)
        video_path = "uploads/" + video_filename
        video_file.save(video_path)

        # Process the video and generate a summary
        output_file = mp42wav(video_path)
        trans = wav2txt(output_file)
        n = chunker(trans)
        result = split_string(trans, n)

        summarized_txt = ''
        for chun_txt in result:
            summarized_txt += summarize_aud(chun_txt)

        return jsonify({"summary": summarized_txt})

if __name__ == "__main__":
    app.run(debug=True)
