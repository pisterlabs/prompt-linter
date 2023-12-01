import openai
import pyttsx3

openai.api_key = 'API_KEY'

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech (words per minute)
    engine.setProperty('volume', 0.8)  # Volume level (0.0 to 1.0)
    engine.say(text)
    engine.runAndWait()

def translate_text(text, source_language, target_language):
    response = openai.Completion.create(
        prompt = f"Translate the following {source_language} text to {target_language}: '{text}'",
        model="text-davinci-003"
        
    )
    return response.choices[0].text

# Example usage
source_text = input("Enter text : ")
source_language = "English"
target_language = input("In which language you want to convert : ")

translated_text = translate_text(source_text, source_language, target_language)
print(f"Translation: {translated_text}")
text_to_speech(translated_text)
