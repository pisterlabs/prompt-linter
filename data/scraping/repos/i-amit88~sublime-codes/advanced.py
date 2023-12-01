import os
import requests
import telebot
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
from dotenv import load_dotenv


from langchain.embeddings import OpenAIEmbeddings

load_dotenv()

SYSTEM_PROMPT = "You are an AI named sonic and you are in a conversation with a human. You can answer questions, provide information, and help with a wide variety of tasks." #os.getenv('SYSTEM_PROMPT')

TELEGRAM_BOT_TOKEN = "6029463044:AAFEBwVEi01UOE7gN8mByhIm9684AKjNEu8"#os.getenv('TELEGRAM_BOT_TOKEN')

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

OPENAI_API_KEY = "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"#os.getenv('OPEN_AI_KEY')

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Store the last 10 conversations for each user
conversations = {}


class LangchainManager:
    def __init__(self):
        self.store = None
        self.vectorstore_info = None
        self.toolkit = None
        self.agent_executor = None
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.llm = OpenAI(temperature=0.7, verbose=True, openai_api_key=OPENAI_API_KEY)

    def getAgentExecutor(self):
        return self.agent_executor

    def getStore(self):
        return self.store

    def upload_pdf(self, path: str):
        # Create and load PDF Loader
        loader = PyPDFLoader(path)
        # Split pages from pdf
        pages = loader.load_and_split()
        self.store = Chroma.from_documents(pages, self.embeddings, collection_name='Pdf')
        # Create vectorstore info object
        self.vectorstore_info = VectorStoreInfo(
            name="Pdf",
            description=" A pdf file to answer your questions",
            vectorstore=self.store
        )
        # Convert the document store into a langchain toolkit
        self.toolkit = VectorStoreToolkit(vectorstore_info=self.vectorstore_info)
        self.agent_executor = create_vectorstore_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            verbose=True
        )

langchain_manager = LangchainManager()

@bot.message_handler(func=lambda message: True, content_types=['document'])
def default_command(message):
    file_info = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open(message.document.file_name, 'wb') as new_file:
        new_file.write(downloaded_file)
    langchain_manager.upload_pdf(message.document.file_name)
    bot.reply_to(message, "File uploaded successfully")
# @app.task
def generate_response_chat(message_list):
    last_message = message_list[-1]
    prompt = last_message #["content"] + "\n\n" + SYSTEM_PROMPT
    response = langchain_manager.getAgentExecutor().run(prompt['content'])
    search = langchain_manager.getStore().similarity_search_with_score(response, k=1)

    assistant_response = search[0][0].page_content#gpt3_response["choices"][0]["message"]["content"].strip()

    return assistant_response


def conversation_tracking(text_message, user_id):
    """
    Make remember all the conversation
    :param old_model: Open AI model
    :param user_id: telegram user id
    :param text_message: text message
    :return: str
    """
    # Get the last 10 conversations and responses for this user
    user_conversations = conversations.get(user_id, {'conversations': [], 'responses': []})
    user_messages = user_conversations['conversations'][-9:] + [text_message]
    user_responses = user_conversations['responses'][-9:]

    # Store the updated conversations and responses for this user
    conversations[user_id] = {'conversations': user_messages, 'responses': user_responses}

    # Construct the full conversation history in the user:assistant, " format
    conversation_history = []

    for i in range(min(len(user_messages), len(user_responses))):
        conversation_history.append({
            "role": "user", "content": user_messages[i]
        })
        conversation_history.append({
            "role": "assistant", "content": user_responses[i]
        })

    # Add last prompt
    conversation_history.append({
        "role": "user", "content": text_message
    })
    # Generate response
    response = generate_response_chat(conversation_history)
    # task = generate_response_chat.apply_async(args=[conversation_history])
    # response = task.get()

    # Add the response to the user's responses
    user_responses.append(response)

    # Store the updated conversations and responses for this user
    conversations[user_id] = {'conversations': user_messages, 'responses': user_responses}

    return response


@bot.message_handler(commands=["start", "help"])
def start(message):
    if message.text.startswith("/help"):
        bot.reply_to(message, "/clear - Clears old "
                              "conversations\nsend text to get replay\nsend voice to do voice"
                              "conversation")
    else:
        bot.reply_to(message, "Just start chatting to the AI or enter /help for other commands")


# Define a function to handle voice messages
# @bot.message_handler(content_types=["voice"])
# def handle_voice(message):
#     user_id = message.chat.id
#     # Download the voice message file from Telegram servers
#     file_info = bot.get_file(message.voice.file_id)
#     file = requests.get("https://api.telegram.org/file/bot{0}/{1}".format(
#         TELEGRAM_BOT_TOKEN, file_info.file_path))
#
#     # Save the file to disk
#     with open("voice_message.ogg", "wb") as f:
#         f.write(file.content)
#
#     # Use pydub to read in the audio file and convert it to WAV format
#     sound = AudioSegment.from_file("voice_message.ogg", format="ogg")
#     sound.export("voice_message.wav", format="wav")
#
#     # Use SpeechRecognition to transcribe the voice message
#     r = sr.Recognizer()
#     with sr.AudioFile("voice_message.wav") as source:
#         # openai.api_key = OPENAI_API_KEY
#         # text = openai.Audio.transcribe("whisper-1", source)
#         # print(text)
#
#         audio_data = r.record(source)
#         text = r.recognize_google(audio_data)
#         print(text)
#
#     # Generate response
#     replay_text = conversation_tracking(text, user_id)
#
#     # Send the question text back to the user
#     # Send the transcribed text back to the user
#     # new_replay_text = "Human: " + text + "\n\n" + "sonic: " + replay_text
#
#     bot.reply_to(message, replay_text)
#
#     # Use Google Text-to-Speech to convert the text to speech
#     tts = gTTS(replay_text)
#     tts.save("voice_message.mp3")
#
#     # Use pydub to convert the MP3 file to the OGG format
#     sound = AudioSegment.from_mp3("voice_message.mp3")
#     sound.export("voice_message_replay.ogg", format="mp3")
#
#     # Send the transcribed text back to the user as a voice
#     voice = open("voice_message_replay.ogg", "rb")
#     bot.send_voice(message.chat.id, voice)
#     voice.close()
#
#     # Delete the temporary files
#     os.remove("voice_message.ogg")
#     os.remove("voice_message.wav")
#     os.remove("voice_message.mp3")
#     os.remove("voice_message_replay.ogg")


@bot.message_handler(func=lambda message: True)
def echo_message(message):
    user_id = message.chat.id

    # Handle /clear command
    if message.text == '/clear':
        conversations[user_id] = {'conversations': [], 'responses': []}
        bot.reply_to(message, "Conversations and responses cleared!")
        return

    response = conversation_tracking(message.text, user_id)

    # Reply to message
    bot.reply_to(message, response)


if __name__ == "__main__":
    print("Starting bot...")
    print("Bot Started")
    print("Press Ctrl + C to stop bot")
    bot.polling(none_stop=True)