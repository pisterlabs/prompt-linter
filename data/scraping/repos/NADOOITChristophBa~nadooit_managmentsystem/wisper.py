import os
import openai

from bot_management.core.service import convert_to_wav

# Get the OpenAI API token from the environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")


def transcribe_audio_file(
    file, supported_formats=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]
):
    # get file extension
    extension = os.path.splitext(file.name)[1][1:]

    file_path = None

    delete_after_transcription = False
    # Convert to wav if the file is oga and not directly supported by Whisper
    if extension not in supported_formats:
        if extension == "oga":
            # Extract the directory and filename without extension
            dir_name, file_name = os.path.split(file.path)
            file_base_name, _ = os.path.splitext(file_name)

            # Generate new file path
            new_file_path = os.path.join(dir_name, file_base_name + ".wav")

            convert_to_wav(file.path, new_file_path)

            file_path = new_file_path

            delete_after_transcription = True
        else:
            # If the file format is not supported by Whisper and not 'oga', raise a ValueError
            raise ValueError("Unsupported file format.")
    else:
        file_path = file.path

    print(file_path)

    # Open the audio file and use Whisper to transcribe it
    with open(file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)

    # If the file was converted to WAV for transcription, delete it after transcription
    if delete_after_transcription:
        os.remove(file_path)

    # Return the transcription
    return transcript["text"]
