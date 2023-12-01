
import openai
import speech_recognition as sr
import pyttsx3
from elevenlabs import generate, play, set_api_key

# Constants
OPENAI_API_KEY = ""
ELEVEN_LABS_API_KEY = ""
VOICE_NAME = "Josh"
MODEL_NAME = "eleven_monolingual_v1"
ACTIVATION_KEYWORD = "computer"
LISTENING_TIMEOUT = 5  # Timeout duration in seconds
VOLUME = 0.7

set_api_key(ELEVEN_LABS_API_KEY)

# GPT-4 interaction
class GPTInteraction:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.engine = pyttsx3.init()
        self.engine.setProperty("volume", VOLUME)  # Set the volume of the text-to-speech output
        self.messages = []

    def start(self):
        self.say("Hello, I'm " + ACTIVATION_KEYWORD + ". How can I assist you?")

        # Set up the OpenAI API credentials
        openai.api_key = OPENAI_API_KEY

        while True:
            try:
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source)
                    print("Listening...")
                    audio = self.recognizer.listen(source, timeout=LISTENING_TIMEOUT)
                    print("Processing audio...")

                recognized_text = self.recognize_speech(audio)
                print("You:", recognized_text)

                activation_index = recognized_text.lower().find(ACTIVATION_KEYWORD)
                if activation_index != -1:
                    user_message = recognized_text[activation_index + len(ACTIVATION_KEYWORD):].strip()
                    
                    if (user_message == "clear history"):
                        self.messages.clear()
                        self.say("Chat history cleared.")
                        continue
                    
                    if (not self.messages):
                        self.messages.append({"role": "system", "content": "You are a helpful assistant."})
                        
                    self.messages.append({"role": "user", "content": user_message})
                    response = self.get_response()
                    self.messages.append({"role": "assistant", "content": response})
                    self.say(response)

            except sr.WaitTimeoutError:
                print("Listening timeout reached. Please try again.")
            except sr.UnknownValueError:
                self.say("Sorry, I didn't catch that. Could you please repeat?")
            except sr.RequestError:
                self.say("Sorry, I'm having trouble accessing the microphone. Please try again.")

    def recognize_speech(self, audio):
        return self.recognizer.recognize_google(audio)

    def get_response(self):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages
        )
        answer = response.choices[0].message.content.strip()
        return answer

    def say(self, text):
        print("GPT:", text)
        
        try:
            audio = generate(
                text=text,
                voice=VOICE_NAME,
                model=MODEL_NAME
                )
 
            play(audio)
        except:
            self.engine.say(text)
            self.engine.runAndWait()
        
        
# Main entry point
if __name__ == "__main__":
    interaction = GPTInteraction()
    interaction.start()
