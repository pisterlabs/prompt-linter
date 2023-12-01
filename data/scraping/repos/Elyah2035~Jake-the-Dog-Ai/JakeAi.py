import openai
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
import simpleaudio as sa

# Replace with your OpenAI API key
api_key = "OpenAiKEY"  # Replace with your actual API key

# Initialize the OpenAI API client
openai.api_key = api_key
triggerword = "Jake"

# Read the character sheet from a file
with open("charactersheet.txt", "r") as file:
    information = file.read()

# Define a function to interact with the chat model
def chat_with_gpt3(prompt, max_tokens=50):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": information},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )
    return response['choices'][0]['message']['content']

# Function to convert text to speech and save it as an MP3 file
def text_to_speech(text, mp3_file):
    tts = gTTS(text, lang='en')
    tts.save(mp3_file)

# Function to play the MP3 file
def play_mp3(mp3_file):
    audio = AudioSegment.from_mp3(mp3_file)
    playback_obj = sa.play_buffer(audio.raw_data, num_channels=audio.channels, bytes_per_sample=audio.sample_width, sample_rate=audio.frame_rate)
    playback_obj.wait_done()

# Function to record speech from the user
def record_user_speech():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Recording now! Speak...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise

        try:
            audio = recognizer.listen(source, timeout=5)  # Record audio for up to 5 seconds
            user_input = recognizer.recognize_google(audio)  # Recognize speech using Google Speech Recognition
            print(f"You (recorded): {user_input}")
            return user_input
        except sr.UnknownValueError:
            print("No speech detected or couldn't understand the audio.")
            return ""
        except sr.RequestError as e:
            print(f"An error occurred during audio recognition: {e}")
            return ""

def trigger():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    alberto = False

    while alberto == False:
        with microphone as source:
            print("Waiting for trigger")
            recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise

            try:
                audio = recognizer.listen(source, timeout=5)  # Record audio for up to 5 seconds
                user_input = recognizer.recognize_google(audio)  # Recognize speech using Google Speech Recognition
                print(f"You (recorded): {user_input}")
            except sr.UnknownValueError:
                print("Waiting again")
                alberto = False
                return False
            except sr.RequestError as e:
                print(f"Sorry, an error occurred during audio recognition: {e}")
                alberto = False
                return False

            if user_input == triggerword:
                print("TRIGGERED")
                alberto = True
                return True

if __name__ == '__main__':
    while True:
        try:
            if trigger():  # Call the trigger function and check its result
                play_mp3("triggersound.mp3")

                user_input = record_user_speech()

                if user_input:
                    response = chat_with_gpt3(user_input)
                    mp3_file = "output2.mp3"

                    # Convert text to speech and save as an MP3
                    text_to_speech(response, mp3_file)

                    # Play the generated MP3
                    play_mp3(mp3_file)

                    print(triggerword + ":")
                    print(response)
        except sr.WaitTimeoutError:
            # Handle the timeout error if no speech input is detected within the specified time
            print("No speech detected during trigger. Waiting again.")
