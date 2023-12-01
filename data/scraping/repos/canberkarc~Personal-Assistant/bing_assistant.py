import openai
import asyncio
import re
import whisper
import boto3
import pydub
from pydub import playback
import speech_recognition as sr
from EdgeGPT import Chatbot, ConversationStyle

# Create a recognizer object and wake word variables
recognizer = sr.Recognizer()
BING_WAKE_WORD = "bing"

    
def synthesize_speech(text, output_filename):
    polly = boto3.client('polly', region_name='us-east-1',  aws_access_key_id='', aws_secret_access_key='')
    
    response = polly.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId='Salli',
        Engine='neural'
    )

    with open(output_filename, 'wb') as f:
        f.write(response['AudioStream'].read())

def play_audio(file):
    sound = pydub.AudioSegment.from_file(file, format="mp3")
    playback.play(sound)

def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # Emojis
                               u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
                               u"\U0001F680-\U0001F6FF"  # Transport & Map Symbols
                               u"\U0001F700-\U0001F77F"  # Alchemical Symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

async def main():

    model = whisper.load_model("base")
    
    synthesize_speech('What can I help you with?', 'C:/Users/a/Desktop/personal_assistant/response.mp3')
    play_audio('C:/Users/a/Desktop/personal_assistant/response.mp3')

    while True:


        with sr.Microphone() as source:
        
            recognizer.adjust_for_ambient_noise(source)

            print("\nSpeak a prompt...")
            audio = recognizer.listen(source)

            try:
                with open("C:/Users/a/Desktop/personal_assistant/audio_prompt.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                
                result = model.transcribe("C:/Users/a/Desktop/personal_assistant/audio_prompt.wav", fp16=False)
                user_input = result["text"]
                print(f"You said: {user_input}")
            except Exception as e:
                print("Error transcribing audio: {0}".format(e))
                continue

            bot = Chatbot(cookiePath='C:/Users/a/Desktop/personal_assistant/cookies.json')
            response = await bot.ask(prompt=user_input, conversation_style=ConversationStyle.precise)
            # Select only the bot response from the response dictionary
            print(response['item'])
            for message in response["item"]["messages"]:
                if message["author"] == "bot":
                    bot_response = message["text"]
            # Remove [^#^] citations in response
            bot_response = re.sub('\[\^\d+\^\]', '', bot_response)
            bot_response = remove_emojis(bot_response)


                
        print("Bot's response:", bot_response)
        synthesize_speech(bot_response, 'C:/Users/a/Desktop/personal_assistant/response.mp3')
        play_audio('C:/Users/a/Desktop/personal_assistant/response.mp3')
        await bot.close()

if __name__ == "__main__":
    asyncio.run(main())