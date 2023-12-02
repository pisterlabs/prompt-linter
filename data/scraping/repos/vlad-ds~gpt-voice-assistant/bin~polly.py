import boto3
from openai import OpenAI

VOICE_ID = "Salli"


def text_to_speech_polly(text: str):
    # Create an Amazon Polly client
    polly = boto3.client('polly')
    
    # Set the output format
    output_format = 'mp3'

    # Set the output file name
    output_file = 'output.mp3'

    # Synthesize speech
    response = polly.synthesize_speech(Text=text, VoiceId=VOICE_ID, OutputFormat=output_format)

    # Save the audio to a file
    with open(output_file, 'wb') as f:
        f.write(response['AudioStream'].read())


def text_to_speech_oa(text: str):
    client = OpenAI()
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text
    )
    response.stream_to_file('output.mp3')
