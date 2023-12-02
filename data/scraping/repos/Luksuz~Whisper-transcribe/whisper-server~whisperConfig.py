import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
apiKey = os.getenv("OPENAI_API_KEY")
openai.api_key = apiKey


def transcribe_audio(audio_file_path):
    try:
        with open(audio_file_path, 'rb') as audio_file:
            transcription = openai.Audio.transcribe("whisper-1", audio_file)
        print(transcription['text'])
        return transcription['text']
    except Exception as e:
        print(f"Error transcribing audio: {str(e)}")
        return None


def abstract_summary_extraction(transcription, purpose):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a highly skilled AI trained in copywriting."
                    "The given text is a speeech-to-text transcription and your task is to rewrite it based on the user puspose provided."
                    "e.g. if the purpose is a birthday letter you have to compile the transcription to make it nicer and consiser, avoiding all of the user speaking imperfections, repetitions and uncertainty while speaking."
                    "Also format it do it fits the purpose, e.g. if the purpose is email, format it so it has a subject and other relevant email fields."
                    "If the user gives an unrealistic or non understandable purpose, return 'unrecognized or invalid purpose given'."
                    "this is the purpose user requested: " + purpose
                },
                {
                    "role": "user",
                    "content": transcription
                }
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error extracting abstract summary: {str(e)}")
        return None


"""# Define other functions (key_points_extraction, action_item_extraction, sentiment_analysis) as needed

if __name__ == "__main__":
    audio_file_path = "your_audio_file.wav"  # Replace with the actual path to your audio file
    transcription = transcribe_audio(audio_file_path)

    if transcription:
        summary = abstract_summary_extraction(transcription)
        if summary:
            print("Abstract Summary:")
            print(summary)
        else:
            print("Failed to extract abstract summary.")
    else:
        print("Failed to transcribe audio.")"""
