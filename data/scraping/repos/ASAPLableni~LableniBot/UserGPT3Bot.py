import numpy as np
import subprocess
import wave
import time

import pandas as pd
import pyaudio
import os
import json
import pickle
# from threading import Thread

# import speech_recognition as sr
import openai
# from contextlib import closing
# from pydub import AudioSegment
from google.cloud import speech

from googletrans import Translator

from boto3 import Session

from pyannote.audio.pipelines import VoiceActivityDetection

from tkinter import *

from Interface import Interface
import utils as ute
from ChatbotGlobals import LableniBot

# ### Interface ###

root = Tk()
root.title("LabLeni BOT")
root.geometry("400x400")

app = Interface(master=root)

app.mainloop()

print(app.bot_config)

subject_id = app.subject_id
print("Subject ID", subject_id)

subject_name = app.subject_name
print("Subject Name", subject_name)

# ### End of the Interface ###

bot_txt, bot_state = app.bot_config.split(" ; ")
bot_txt_to_root, bot_state_to_root = bot_txt.replace(" ", "_"), bot_state.replace(" ", "_")

# ######################################
# ### Opening PARAMETERS CONFIG file ###
# ######################################

# ### Initial message to de chatbot ###

HUMAN_START_SEQUENCE = subject_name + ":"

# ############################
# ### Open Parameters dict ###
# ############################

with open("LableniBotConfig/bot_parameters.json", "r", encoding='utf-8') as read_file:
    parameters_dict = json.load(read_file)

TRANSLATION_MODULE = parameters_dict["TRANSLATION_MODULE"]
OMNIVERSE_MODULE = parameters_dict["OMNIVERSE_MODULE"]

INITIAL_TOKENS_OPENAI = parameters_dict["INITIAL_TOKENS_OPENAI"]
BOT_MODEL = parameters_dict["BOT_MODEL"]
BOT_TEMPERATURE = parameters_dict["BOT_TEMPERATURE"]
BOT_FREQUENCY_PENALTY = parameters_dict["BOT_FREQUENCY_PENALTY"]
BOT_PRESENCE_PENALTY = parameters_dict["BOT_PRESENCE_PENALTY"]

ACTIVATE_MAX_TIME_TH = parameters_dict["ACTIVATE_MAX_TIME_TH"]
MAX_TIME_TH_s = parameters_dict["MAX_TIME_TH_s"]

# Time to wait until ask the user to repeat.
waitTime = 15
# Audio record parameters.
CHUNK = parameters_dict["CHUNK"]
FORMAT = pyaudio.paInt16
CHANNELS = parameters_dict["CHANNELS"]
RATE = parameters_dict["RATE"]
RECORD_SECONDS = parameters_dict["RECORD_SECONDS"]
TIME_TO_CUT = parameters_dict["TIME_TO_CUT"]

# ###########################
# ### Opening CONFIG file ###
# ###########################

config_json = open("LableniBotConfig/config.json")
config_dict = json.load(config_json)

session = Session(
    aws_access_key_id=config_dict["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=config_dict["AWS_SECRET_ACCESS_KEY"]
)
polly = session.client("polly", region_name='eu-west-1')
openai.api_key = config_dict["OPENAI_KEY"]

# ###########################
# ### Opening GOOGLE file ###
# ###########################

GOOGLE_ROOT = config_dict["GOOGLE_ROOT"]
google_client = speech.SpeechClient.from_service_account_json(GOOGLE_ROOT)

# ###############################
# ### SILENCE DETECTION MODEL ###
# ###############################

hugging_face_key = config_dict["HUGGING_FACE"]
silence_detection_pipeline = VoiceActivityDetection(segmentation="pyannote/segmentation",
                                                    use_auth_token=hugging_face_key)

HYPER_PARAMETERS = {
    # onset/offset activation thresholds
    "onset": 0.5, "offset": 0.5,
    # remove speech regions shorter than that many seconds.
    "min_duration_on": 0.0,
    # fill non-speech regions shorter than that many seconds.
    "min_duration_off": 0.0
}
silence_detection_pipeline.instantiate(HYPER_PARAMETERS)
_ = silence_detection_pipeline("audio_bot_aws.wav")

# ##########################
# ### TRANSLATION MODULE ###
# ##########################

# if TRANSLATION_MODULE:
# Initialize the translator model
google_translator = Translator()

# #################
# ### CONSTANTS ###
# #################

OUTPUT_FILE_IN_WAVE = "audio_bot_aws.wav"  # WAV format Output file  name

# Modes avaible: 'voice' or 'write'.
CHAT_MODE = "voice"  # TODO: Put the parameter in the interface.

# ########################
# ### OMNIVERSE MODULE ###
# ########################

if OMNIVERSE_MODULE:
    ROOT_TO_OMNIVERSE = config_dict["ROOT_TO_OMNIVERSE"]

# ##############
# ### INPUTS ###
# ##############

# Begin of the session
PATH_TO_DATA = "Conversations/" + subject_id
if not os.path.exists(PATH_TO_DATA):
    os.mkdir(PATH_TO_DATA)

init_time = ute.get_current_time(get_time_str=True)
init_str_time, unix_time = ute.get_current_time()
SUB_PATH_TO_DATA = PATH_TO_DATA + "/" + subject_id + "_" + init_time
os.mkdir(SUB_PATH_TO_DATA)

# Create Audios folder
os.mkdir(SUB_PATH_TO_DATA + "/Audios")
WAVE_OUTPUT_FILENAME = SUB_PATH_TO_DATA + "/Audios/Subject_" + subject_id
os.mkdir(SUB_PATH_TO_DATA + "/BotAudios")
WAVE_OUTPUT_FILENAME_BOT = SUB_PATH_TO_DATA + "/BotAudios/BotSubject_" + subject_id


# ### Call the time ###
# subprocess.call("python clock_track.py")

def init_clock():
    os.system("python clock_track.py")


# Thread(target=init_clock).start()

# ################################
# ### DEFINE THE CLASS CHATBOT ###
##################################

my_chatbot = LableniBot(
    subject_id=subject_id,
    subject_name=subject_name,
    mode_chat=CHAT_MODE,
    path_to_bot_param="LableniBotConfig/bot_parameters.json",
    path_to_bot_personality="LableniBotConfig/Personalities/" + bot_txt_to_root + "/" + bot_state_to_root + ".json",
    path_to_save=SUB_PATH_TO_DATA + "/Conv_" + str(init_time),
)

# ### Change the name of the avatar if it is needed ###

if app.change_avatar_name:
    my_chatbot.global_message = my_chatbot.global_message.replace(my_chatbot.bot_name, app.avatar_name)
    my_chatbot.initial_message = my_chatbot.initial_message.replace(my_chatbot.bot_name, app.avatar_name)

    my_chatbot.bot_name = app.avatar_name
    my_chatbot.bot_start_sequence = my_chatbot.bot_name + ":"

if os.path.exists("Conversations/" + subject_id + '/GuideOfTimes.pkl'):
    with open("Conversations/" + subject_id + '/GuideOfTimes.pkl', 'rb') as f:
        guide_of_times = pickle.load(f)

    guide_of_times.append({
        "RealTimeStr": init_str_time,
        "UnixTime": unix_time,
        "Event": my_chatbot.config_name + "_start"
    })
else:
    guide_of_times = [{
        "RealTimeStr": init_str_time,
        "UnixTime": unix_time,
        "Event": my_chatbot.config_name + "_start"
    }]

with open("Conversations/" + subject_id + '/GuideOfTimes.pkl', 'wb') as handle:
    pickle.dump(guide_of_times, handle, protocol=pickle.HIGHEST_PROTOCOL)

bot_result_list = []
spanish_text = " "
repeat_message_label = False
t0_init_pipeline = time.time()
try:
    while True:

        # #############################
        # ### BOT GENERATE SENTENCE ###
        # #############################

        t_str_loop_start, t_unix_loop_start = ute.get_current_time()
        if time.time() - t0_init_pipeline > MAX_TIME_TH_s and ACTIVATE_MAX_TIME_TH:
            t_i_openai = ute.get_current_time(only_unix=True)

            bot_answer = my_chatbot.farewell_message

            my_chatbot.good_bye_message = True
            t_f_openai = ute.get_current_time(only_unix=True)

        elif spanish_text is not None and not repeat_message_label:
            if my_chatbot.counter_conv_id > 0:

                t_i_openai = ute.get_current_time(only_unix=True)

                response = openai.Completion.create(
                    engine=BOT_MODEL,
                    prompt=my_chatbot.global_message,
                    temperature=BOT_TEMPERATURE,
                    max_tokens=INITIAL_TOKENS_OPENAI,
                    top_p=1,
                    frequency_penalty=BOT_FREQUENCY_PENALTY,
                    presence_penalty=BOT_PRESENCE_PENALTY,
                    stop=["Humano:", "Michelle:", subject_name + ":"]
                )

                t_f_openai = ute.get_current_time(only_unix=True)

                bot_answer = response["choices"][0]["text"]
                if len(bot_answer.split(":")) > 1:
                    bot_answer = my_chatbot.remove_spanish_accents(bot_answer)
                    if subject_name + ":" in bot_answer:
                        bot_answer = bot_answer.split(subject_name + ":")[0]

            else:
                t_i_openai = ute.get_current_time(only_unix=True)
                bot_answer = my_chatbot.initial_message
                t_f_openai = ute.get_current_time(only_unix=True)
        else:
            t_i_openai = ute.get_current_time(only_unix=True)
            bot_answer = my_chatbot.sentence_to_repeat
            t_f_openai = ute.get_current_time(only_unix=True)
            repeat_message_label = False

        # Clean chatbot message
        bot_message = my_chatbot.clean_bot_message(bot_answer)

        try:
            detect_language = google_translator.detect(bot_message)
            if detect_language.lang != "es":
                x = google_translator.translate(bot_message, dest=my_chatbot.native_language)
                bot_message_filtered = x.text
            else:
                bot_message_filtered = bot_message
        except Exception as ex:
            bot_message_filtered = bot_message

        my_chatbot.global_message += bot_message_filtered
        print("*** Global message *** \n", my_chatbot.global_message)

        # ##########################
        # ### AWS TEXT TO SPEECH ###
        # ##########################

        t_i_aws = ute.get_current_time(only_unix=True)

        # Message string transform to AWS Polly PCM format.
        bot_message_spanish_aws = my_chatbot.from_str_to_aws_polly_pcm(bot_message_filtered)

        RATE_AWS = 16000  # Polly supports 16000Hz and 8000Hz output for PCM format
        response = polly.synthesize_speech(
            Text=bot_message_spanish_aws,
            OutputFormat="pcm",
            VoiceId=my_chatbot.bot_voice_id,
            # SampleRate=str(RATE),
            Engine=my_chatbot.engine_type,
            TextType="ssml"
        )

        t_f_aws = ute.get_current_time(only_unix=True)

        # ##############################
        # ### LISTEN AVATAR SENTENCE ###
        # ##############################

        # The following code will save the 'response' of AWS polly in a '.wav' format.

        # Initializing variables
        OMNI_CHANNELS = 1  # Polly's output is a mono audio stream
        WAV_SAMPLE_WIDTH_BYTES = 2  # Polly's output is a stream of 16-bits (2 bytes) samples
        FRAMES = []

        # Processing the response to audio stream
        STREAM = response.get("AudioStream")
        FRAMES.append(STREAM.read())
        # TODO: Save these audios as the subjects audios are saved.
        root_bot_audio = ROOT_TO_OMNIVERSE + "/" + OUTPUT_FILE_IN_WAVE if OMNIVERSE_MODULE else OUTPUT_FILE_IN_WAVE

        WAVE_FORMAT = wave.open(root_bot_audio, 'wb')
        WAVE_FORMAT.setnchannels(OMNI_CHANNELS)
        WAVE_FORMAT.setsampwidth(WAV_SAMPLE_WIDTH_BYTES)
        WAVE_FORMAT.setframerate(RATE_AWS)
        WAVE_FORMAT.writeframes(b''.join(FRAMES))
        WAVE_FORMAT.close()

        t_bot_talk_start = ute.get_current_time(only_unix=True)

        if OMNIVERSE_MODULE:

            # #################
            # ### OMNIVERSE ###
            # #################

            # OMNIVERSE_AVATAR = "/Woman/audio_player_streaming"
            call_to_omniverse = " python " + ROOT_TO_OMNIVERSE + "/my_test_client.py "
            call_to_omniverse += " " + ROOT_TO_OMNIVERSE + "/" + OUTPUT_FILE_IN_WAVE
            call_to_omniverse += " " + my_chatbot.omniverse_avatar
            # call_to_omniverse += " " + OUTPUT_FILE_IN_WAVE.replace(".wav", ".mp3") + " " + OMNIVERSE_AVATAR
            subprocess.call(call_to_omniverse, shell=True)
        else:
            ute.reproduce_audio(root_bot_audio, CHUNK)

        t_bot_talk_end = ute.get_current_time(only_unix=True)

        # #######################
        # ### APPEND BOT DATA ###
        # #######################

        my_chatbot.append_data(
            t_str_loop_start=t_str_loop_start, t_unix_loop_start=t_unix_loop_start,
            source="Bot", source_message=bot_message_filtered,
            bot_start_unix=t_i_openai, bot_end_unix=t_f_openai,
            aws_start_unix=t_i_aws, aws_end_unix=t_f_aws,
            s2t_start_unix=np.nan, s2t_end_unix=np.nan,
            bot_talk_start_unix=t_bot_talk_start, bot_talk_end_unix=t_bot_talk_end,
            person_talk_start_unix=np.nan, person_talk_end_unix=np.nan
        )
        my_chatbot.save_data()

        # ##################
        # ### HUMAN TALK ###
        # ##################

        if CHAT_MODE == "voice":

            t_person_talk_start = ute.get_current_time(only_unix=True)

            t0 = time.time()
            t0_start_talk = time.time()
            silence_th = 0

            p = pyaudio.PyAudio()
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

            print("*** Recording ***")

            frames = []

            wf = wave.open(WAVE_OUTPUT_FILENAME + "_T=" + str(my_chatbot.counter_conv_id) + ".wav", 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)  # , exception_on_overflow=False
                frames.append(data)

                # ################################
                # ### SILENCE DETECTION MODULE ###
                # ################################
                if time.time() - t0 > 3:
                    wf = wave.open(WAVE_OUTPUT_FILENAME + "_T=" + str(my_chatbot.counter_conv_id) + ".wav", 'wb')
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(p.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(frames))
                    wf.close()

                    vad = silence_detection_pipeline(
                        WAVE_OUTPUT_FILENAME + "_T=" + str(my_chatbot.counter_conv_id) + ".wav"
                    )

                    x = vad.get_timeline().segments_set_

                    if len(x) > 0:
                        last_time_talk = np.max([x_elt.end for x_elt in list(x)])
                        cond_listen = (time.time() - t0_start_talk) > 5
                        if time.time() - (last_time_talk + t0_start_talk) > TIME_TO_CUT and cond_listen:
                            break
                        else:
                            silence_th += len(x) - 1

                    t0 = time.time()

            print("*** Done recording ***")

            stream.stop_stream()
            stream.close()
            p.terminate()

            # Save the audio of the subject.
            subject_speech_file = WAVE_OUTPUT_FILENAME + "_T=" + str(my_chatbot.counter_conv_id) + ".wav"
            wf = wave.open(subject_speech_file, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            t_person_talk_end = ute.get_current_time(only_unix=True)

            t_i_s2t = ute.get_current_time(only_unix=True)

            # #############################
            # ### GOOGLE SPEECH TO TEXT ###
            # #############################

            with open(subject_speech_file, "rb") as audio_file:
                content = audio_file.read()
            audio = speech.RecognitionAudio(content=content)

            google_config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                # sample_rate_hertz=44100,
                language_code=my_chatbot.native_language + "-EU",
                audio_channel_count=1,
                enable_separate_recognition_per_channel=True,
                enable_automatic_punctuation=True
            )
            response = google_client.recognize(config=google_config, audio=audio)

            t_f_s2t = ute.get_current_time(only_unix=True)

            spanish_text, repeat_message_label = ute.process_googles2t_answer(response.results)
            spanish_text = my_chatbot.remove_spanish_accents(spanish_text)

        elif CHAT_MODE == "write":
            print("Write something....")
            t_i_s2t, t_f_s2t = np.nan, np.nan
            t_person_talk_start = ute.get_current_time(only_unix=True)
            spanish_text = input()
            t_person_talk_end = ute.get_current_time(only_unix=True)
        else:
            print("Please select between 'write' or 'voice' method")
            break

        # ###################
        # ### TRANSLATION ###
        # ###################

        if TRANSLATION_MODULE:
            # Here the message is translated from spanish to english.
            x = google_translator.translate(spanish_text)
            person_message = x.text + " ."
        else:
            person_message = spanish_text

        person_message = person_message if person_message[-1] in [".", "?", "!"] else person_message + "."
        my_chatbot.global_message += "\n" + HUMAN_START_SEQUENCE + " " + person_message

        my_chatbot.global_message += "\n" + my_chatbot.bot_start_sequence + " "

        my_chatbot.append_data(
            t_str_loop_start=np.nan, t_unix_loop_start=np.nan,
            source="Person", source_message=spanish_text,
            bot_start_unix=np.nan, bot_end_unix=np.nan,
            aws_start_unix=np.nan, aws_end_unix=np.nan,
            s2t_start_unix=t_i_s2t, s2t_end_unix=t_f_s2t,
            bot_talk_start_unix=np.nan, bot_talk_end_unix=np.nan,
            person_talk_start_unix=t_person_talk_start, person_talk_end_unix=t_person_talk_end
        )
        my_chatbot.save_data()

        print("*** Data saved *** ")

        # ##################
        # ### COUNTER ID ###
        # ##################
        # This counter identifies the iteration of the conversation. If Bot and Human have talked,
        # another iteration starts.
        my_chatbot.counter_conv_id += 1

        if my_chatbot.good_bye_message:
            my_chatbot.save_guide_of_times()
            my_chatbot.from_pkl_to_csv()

            break

except KeyboardInterrupt:

    my_chatbot.save_guide_of_times()
    my_chatbot.from_pkl_to_csv()
