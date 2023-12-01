import openai
import pyttsx3

# Set your OpenAI API key
openai.api_key = "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"
MALE_VOICE_TOKEN = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0"
FEMALE_VOICE_TOKEN = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"






class Chatbot:
    def __init__(self):
        self.engine = pyttsx3.init()

    def text_to_speech(self, text, gender):
        if gender == "m":
            self.engine.setProperty('voice', "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0")
        elif gender == "f":
            self.engine.setProperty('voice', "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0")

        # Set the speech rate to be 10% faster than the default rate
        self.engine.setProperty('rate', self.engine.getProperty('rate') * 1.1)

        # Speak the provided text
        self.engine.say(text)
        self.engine.runAndWait()


class AIChatbot:
    def __init__(self):
        self.chatbot = Chatbot()
        self.conversation = []

    def get_chatbot_response(self, user_input):
        role = "user"
        content = user_input

        # Create completion
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": role, "content": content}
            ]
        )

        # Return the response from the completion
        return completion.choices[0].message.content

    def run_chatbot(self):
        user_input = "HI! welcome to the AI podcast, we are both AI and we will discuss subjects. What do you want to talk about?"
        response = self.get_chatbot_response(user_input)
        self.conversation.append(response)

        while True:
            response = self.get_chatbot_response(self.conversation[-1])
            self.conversation.append(response)
            print(response)

            # Determine gender based on conversation length
            if len(self.conversation) % 2 == 0:
                gender = "m"
            else:
                gender = "f"


            # Use text-to-speech to speak the response
            self.chatbot.text_to_speech(response, gender)





ai_chatbot = AIChatbot()
ai_chatbot.run_chatbot()
# Replace 'FaceG_talk.png' and 'FaceM_talk.png' with the actual paths to your images
