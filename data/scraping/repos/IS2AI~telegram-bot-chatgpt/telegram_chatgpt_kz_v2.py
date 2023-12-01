from espnet2.bin.tts_inference import Text2Speech
from googletrans import Translator
from issai.asr import ASR
from issai import utils
import openai
import subprocess
import soundfile
import telebot
import json
import os


# create directories to save
# input and output files
utils.make_dir("input/voice")
utils.make_dir("output")

# initialize telegram bot
isRunning = False
tele_token = "YOUR TELEGRAM BOT TOKEN"
tele_bot = telebot.TeleBot(tele_token, threaded=True)

# the first message to ChatGPT
messages = [
    # system message first, it helps set the behavior of the assistant
    {"role": "system", "content": "You are a helpful assistant."},
]
reset_chat = True # set False if you want ChatGPT to use chat history

# initialize translator
translator = Translator()

# initialize ASR model
asr = ASR(lang='kk', model='google') # to use offline vosk asr: 'google' -> 'vosk'

# define path to the tts model weights
model_path = "exp/tts_train_raw_char/train.loss.ave_5best.pth"
vocoder_path = "exp/checkpoint-400000steps.pkl"

# load the kazakh tts model
tts = Text2Speech.from_pretrained(
    model_file=model_path,
    vocoder_file=vocoder_path,
    device="cpu",
    threshold=0.5,
    minlenratio=0.0,
    maxlenratio=10.0,
    use_att_constraint=False,
    backward_window=1,
    forward_window=3,
    prefer_normalized_feats=True)

@tele_bot.message_handler(commands=["start", "go"])
def start_handler(message):
    global isRunning
    if not isRunning:
        tele_bot.send_message(message.chat.id, "Дауыс ChatGPT-ға қош келдіңіз!")
        isRunning = True
        #TODO
        #Send an audio welcome message

@tele_bot.message_handler(content_types=['voice'])
def voice_processing(message):
    # process the voice message
    file_info = tele_bot.get_file(message.voice.file_id)
    file_data = tele_bot.download_file(file_info.file_path)

    raw_audio_path = './input/' + file_info.file_path
    with open(raw_audio_path, 'wb') as f:
        f.write(file_data)

    #tele_bot.reply_to(message, "Processing...")
    input_audio_path = raw_audio_path + ".wav"
    process = subprocess.run(['ffmpeg', '-i', raw_audio_path, input_audio_path])
    if process.returncode != 0:
        raise Exception("Something went wrong")

    # convert the audio input to text
    asr.convert(input_audio_path)
    tele_bot.reply_to(message, "Сіз: " + asr.message)
    print("User:", asr.message)

    # translate to english
    message_tr = translator.translate(asr.message, src='kk', dest='en')
    message_tr = message_tr.text

    print("User (kk):", asr.message)
    print("User (en):", message_tr)

    # send the message to ChatGPT
    if reset_chat:
        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": message_tr}])
    else:
        messages.append({"role": "user", "content": message_tr},)
        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages)
        messages.append({"role": "assistant", "content": chat_completion.choices[0].message.content})

    response = chat_completion.choices[0].message.content
    print("ChatGPT (en):", response)

    # translate the response to english
    response = translator.translate(response, src='en', dest='kk')
    print("ChatGPT (kk):", response.text)
    response_orig = response.text

    # sent the text response to telegram
    tele_bot.send_message(message.chat.id, response_orig)

    # preprocess the prompt for espnet
    response = utils.preprocess_text(response.text)

    # create a path to save the audio
    output_audio_path = os.path.join('output', 'answer.wav')

    # convert the answer to speech
    wav = tts(response)["wav"]

    # save the generated speech
    soundfile.write(output_audio_path, wav.numpy(), tts.fs, "PCM_16")

    # send the voice response to the telegram
    tele_bot.send_voice(message.chat.id, voice=open(output_audio_path, "rb"))

    # remove input and output files
    os.remove(raw_audio_path)
    os.remove(input_audio_path)
    os.remove(output_audio_path)

# run the telegram bot
tele_bot.polling(none_stop=True)
