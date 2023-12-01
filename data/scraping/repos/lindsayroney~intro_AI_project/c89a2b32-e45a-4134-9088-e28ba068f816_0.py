import openai
import command_manager
import voice_feedback

bot = None  # the ChatGPT bot object
openai.api_key = 'your-api-key'  # replace 'your-api-key' with your actual OpenAI API key

def chat(text):
    """
    handles user-chatgpt interactions
    """
    if command_manager.hasText(text, command_manager.deactivateChatMode):
        voice_feedback.speak('deactivating chatgpt mode', wait=True)
        command_manager.chatMode = False
        return
    global bot
    if not bot:
        try:
            bot = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": text}
                ]
            )
        except Exception as e:
            print(e)
    print(f"You to ChatGPT: {text}")
    response = bot['choices'][0]['message']['content']
    voice_feedback.speak(response, wait=True)
