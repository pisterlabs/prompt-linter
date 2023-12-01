import os
import openai
from flask import Flask, request, jsonify

app = Flask(__name__)

# Set your OpenAI API key here
openai.api_key = "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"

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        audio_file = request.files['audio']
        language = request.form['language']

        if not audio_file or not language:
            return jsonify({"error": "Missing audio file or language parameter"}), 400

        # Save the audio file
        audio_filename = "uploaded_audio.wav"  # Change the extension based on the actual format
        audio_file.save(audio_filename)

        # Read the audio file as binary data
        with open(audio_filename, "rb") as audio_file:
            transcript = openai.Audio.transcribe(
                file=audio_file,
                model="whisper-1",
                response_format="text",
                language=language
            )

        os.remove(audio_filename)  # Remove the temporary audio file

        return jsonify({"transcript": transcript})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
