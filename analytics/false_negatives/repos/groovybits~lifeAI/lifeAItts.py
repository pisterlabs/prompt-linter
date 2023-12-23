#!/usr/bin/env python

## Life AI Text to Speech module
#
# Chris Kennedy 2023 (C) GPL
#
# Free to use for any use as in truly free software
# as Richard Stallman intended it to be.
#

import zmq
import argparse
import requests
import io
import warnings
import re
import logging
import time
import os
from dotenv import load_dotenv
import inflect
import traceback
import soundfile as sf
import torch
from transformers import VitsModel, AutoTokenizer
from transformers import logging as trlogging
from pydub import AudioSegment
import gender_guesser.detector as gender_guess
from openai import OpenAI
import json

trlogging.set_verbosity_error()

load_dotenv()

# Suppress warnings
warnings.simplefilter(action='ignore', category=Warning)

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove image tags or Markdown image syntax
    text = re.sub(r'\!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'<img.*?>', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove any inline code blocks
    text = re.sub(r'`.*?`', '', text)
    
    # Remove any block code segments
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    
    # Remove special characters and digits (optional, be cautious)
    #text = re.sub(r'[^a-zA-Z0-9\s.?,!\n:\'\"\-\t]', '', text)

    if args.service == "mms-tts":
        p = inflect.engine()

        def num_to_words(match):
            number = match.group()
            try:
                words = p.number_to_words(number)
            except inflect.NumOutOfRangeError:
                words = "[number too large]"
            return words

        text = re.sub(r'\b\d+(\.\d+)?\b', num_to_words, text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())

    return text

def get_aac_duration(aac_data):
    audio_segment = AudioSegment.from_file(io.BytesIO(aac_data), format='aac')
    return len(audio_segment) / 1000.0  # Convert from milliseconds to seconds

def get_tts_audio(service, text, voice=None, noise_scale=None, noise_w=None, length_scale=None, ssml=None, audio_target=None):
    
    if service == "mimic3":
        params = {
            'text': text,
            'voice': voice or 'en_US/cmu-arctic_low#slt',
            'noiseScale': noise_scale or '0.333',
            'noiseW': noise_w or '0.333',
            'lengthScale': length_scale or '1.5',
            'ssml': ssml or 'false',
            'audioTarget': audio_target or 'client'
        }

        response = requests.get('http://earth:59125/api/tts', params=params)
        response.raise_for_status()
        return response.content
    elif service == "openai":
        response = openai_client.audio.speech.create(
            model='tts-1',
            voice= voice or 'nova',
            input=text,
            speed=length_scale or '1.0',
            response_format='aac'
        )

        return response.content
    elif service == "mms-tts":
        inputs = tokenizer(text, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'].long()

        output = None
        try:
            with torch.no_grad():
                output = model(**inputs).waveform
            waveform_np = output.squeeze().numpy().T
        except Exception as e:
            logger.error(f"{traceback.print_exc()}")
            logger.error(f"Exception: ERROR STT error with output.squeeze().numpy().T on audio: {text}")
            return None
        
        audiobuf = io.BytesIO()
        sf.write(audiobuf, waveform_np, model.config.sampling_rate, format='WAV')
        audiobuf.seek(0)
        
        return audiobuf.getvalue()

def main():
    voice_set = False
    gender = args.gender
    last_gender = gender
    voice_model = args.voice
    last_voice_model = voice_model
    male_voice_index = 0
    female_voice_index = 0
    speaker_map = {}
    speaker = "narrator"
    last_speaker = speaker
    mediaid = 0
    last_mediaid = mediaid
    # voice, gender
    male_voices = []
    female_voices = []
    speaker_count = 0

    tts_api = args.service
    last_tts_api = "" # start off with no last TTS API so initializes
    voice_speed = "1.5" # default speed for mimic3, else they are too fast (higher slower)

    speaker_map[speaker] = {"gender": gender, "voice": voice_model}

    while True:
        header_message = receiver.recv_json()
        segment_number = header_message["segment_number"]
        text = header_message["text"]
        episode_msg = header_message["episode"]
        mediaid = header_message["mediaid"]

        is_episode = False
        if episode_msg == "true":
            is_episode = True
        else:
            is_episode = False

        if last_mediaid != mediaid:
            last_mediaid = mediaid
            voice_model = args.voice
            last_voice_model = voice_model
            gender = args.gender
            last_gender = gender
            male_voice_index = 0
            female_voice_index = 0
            speaker = "narrator"
            last_speaker = speaker
            tts_api = args.service

            ## set the defaults
            voice_model = None
            voice_speed = "1.5"
            speaker_count = 0

        header_voice_model = None
        if 'voice_model' in header_message:
            header_voice_model = header_message["voice_model"]
            tts_api = header_voice_model.split(":")[0]
            logger.info(f"Text to Speech: Header voice model {header_voice_model}.")

        # switched TTS Services
        if tts_api != last_tts_api:
            last_tts_api = tts_api
            speaker_map = {} # reset the speaker map
            speaker_map[speaker] = {"gender": gender, "voice": voice_model}
            # OpenAI API
            if tts_api == "openai":
                male_voices = ['alloy', 'echo', 'fabel', 'oynx']
                female_voices = ['nova', 'shimmer']
                default_voice = 'nova'
                voice_model = default_voice
            elif tts_api == "mimic3":
                male_voices = [
                    #'en_US/hifi-tts_low#6097',
                    #'en_US/hifi-tts_low#9017',
                    'en_US/vctk_low#p326',
                    'en_US/vctk_low#p259',
                    'en_US/vctk_low#p247',
                    'en_US/vctk_low#p263',
                    'en_US/vctk_low#p286',
                    'en_US/vctk_low#p270',
                    'en_US/vctk_low#p281',
                    'en_US/vctk_low#p271',
                    'en_US/vctk_low#p273',
                    'en_US/vctk_low#p284',
                    'en_US/vctk_low#p287',
                    'en_US/vctk_low#p360',
                    'en_US/vctk_low#p376',
                    'en_US/vctk_low#p304',
                    'en_US/vctk_low#p347',
                    'en_US/vctk_low#p311',
                    'en_US/vctk_low#p334',
                    'en_US/vctk_low#p316',
                    'en_US/vctk_low#p363',
                    'en_US/vctk_low#p275',
                    'en_US/vctk_low#p258',
                    'en_US/vctk_low#p232',
                    'en_US/vctk_low#p292',
                    'en_US/vctk_low#p272',
                    'en_US/vctk_low#p278',
                    'en_US/vctk_low#p298',
                    'en_US/vctk_low#p279',
                    'en_US/vctk_low#p285',
                    'en_US/vctk_low#p326', # super deep voice
                    'en_US/vctk_low#p254',
                    'en_US/vctk_low#p252',
                    'en_US/vctk_low#p345',
                    'en_US/vctk_low#p243',
                    'en_US/vctk_low#p227',
                    'en_US/vctk_low#p225',
                    'en_US/vctk_low#p251',
                    'en_US/vctk_low#p246',
                    'en_US/vctk_low#p226',
                    'en_US/vctk_low#p260',
                    'en_US/vctk_low#p245',
                    'en_US/vctk_low#p241',
                    'en_US/vctk_low#p237',
                    'en_US/vctk_low#p256',
                    'en_US/vctk_low#p302',
                    'en_US/vctk_low#p264',
                    'en_US/vctk_low#p225',
                    'en_US/cmu-arctic_low#rms',
                    'en_US/cmu-arctic_low#ksp',
                    'en_US/cmu-arctic_low#aew',
                    'en_US/cmu-arctic_low#bdl',
                    'en_US/cmu-arctic_low#jmk',
                    'en_US/cmu-arctic_low#fem',
                    'en_US/cmu-arctic_low#ahw',
                    'en_US/cmu-arctic_low#aup',
                    'en_US/cmu-arctic_low#gke'
                ]
                female_voices = [
                    'en_US/vctk_low#p303', #'en_US/hifi-tts_low#92',
                    'en_US/vctk_low#s5',
                    'en_US/vctk_low#p264',
                    'en_US/vctk_low#p239',
                    'en_US/vctk_low#p236',
                    'en_US/vctk_low#p250',
                    'en_US/vctk_low#p261',
                    'en_US/vctk_low#p283',
                    'en_US/vctk_low#p276',
                    'en_US/vctk_low#p277',
                    'en_US/vctk_low#p231',
                    'en_US/vctk_low#p238',
                    'en_US/vctk_low#p257',
                    'en_US/vctk_low#p329',
                    'en_US/vctk_low#p261',
                    'en_US/vctk_low#p310',
                    'en_US/vctk_low#p340',
                    'en_US/vctk_low#p330',
                    'en_US/vctk_low#p308',
                    'en_US/vctk_low#p314',
                    'en_US/vctk_low#p317',
                    'en_US/vctk_low#p339',
                    'en_US/vctk_low#p294',
                    'en_US/vctk_low#p305',
                    'en_US/vctk_low#p266',
                    'en_US/vctk_low#p318',
                    'en_US/vctk_low#p323',
                    'en_US/vctk_low#p351',
                    'en_US/vctk_low#p333',
                    'en_US/vctk_low#p313',
                    'en_US/vctk_low#p244',
                    'en_US/vctk_low#p307',
                    'en_US/vctk_low#p336',
                    'en_US/vctk_low#p312',
                    'en_US/vctk_low#p267',
                    'en_US/vctk_low#p297',
                    'en_US/vctk_low#p295',
                    'en_US/vctk_low#p288',
                    'en_US/vctk_low#p301',
                    'en_US/vctk_low#p280',
                    'en_US/vctk_low#p241',
                    'en_US/vctk_low#p268',
                    'en_US/vctk_low#p299',
                    'en_US/vctk_low#p300',
                    'en_US/vctk_low#p230',
                    'en_US/vctk_low#p269',
                    'en_US/vctk_low#p293',
                    'en_US/vctk_low#p262',
                    'en_US/vctk_low#p343',
                    'en_US/vctk_low#p229',
                    'en_US/vctk_low#p240',
                    'en_US/vctk_low#p248',
                    'en_US/vctk_low#p253',
                    'en_US/vctk_low#p233',
                    'en_US/vctk_low#p228',
                    'en_US/vctk_low#p282',
                    'en_US/vctk_low#p234',
                    'en_US/vctk_low#p303', # nice crackly voice
                    'en_US/vctk_low#p265',
                    'en_US/vctk_low#p306',
                    'en_US/vctk_low#p249',
                    'en_US/vctk_low#p362',
                    'en_US/ljspeech_low',
                    'en_US/cmu-arctic_low#ljm',
                    'en_US/cmu-arctic_low#slp',
                    'en_US/cmu-arctic_low#axp',
                    'en_US/cmu-arctic_low#eey',
                    'en_US/cmu-arctic_low#lnh',
                    'en_US/cmu-arctic_low#elb',
                    'en_US/cmu-arctic_low#slt'
                ]
                default_voice = 'en_US/vctk_low#p303',
                voice_model = default_voice

        # request to change the gender
        gender_custom = None
        gender_override = False
        if 'gender' in header_message:
            gender_custom = header_message['gender']
            gender_override = True

        voice_model_custom = None
        voice_speed_custom = None
        voice_override = False
        # request to switch the voice model
        if header_voice_model:
            voice_data = header_voice_model
            # "voice_model": "mimic3:en_US/cmu-arctic_low#eey:1.2",
            # TTS API, Voice Model to use, Voice Model Speed to use
            tts_api = voice_data.split(":")[0]
            voice_model_custom = voice_data.split(":")[1]
            voice_speed_custom = voice_data.split(":")[2]
            voice_override = True  # only set voice if it's not an episode
            logger.info(
                f"Text to Speech: Custom settings {voice_model_custom} at speed {voice_speed_custom} using API {tts_api}.")

            # Custom speaker name
            ainame = header_message["ainame"].lower()

            # add custom voice to speaker map
            if ainame not in speaker_map:
                mapped_gender = last_gender
                if gender_override:
                    mapped_gender = gender_custom
                speaker_map[ainame] = {
                    "gender": mapped_gender, "voice": voice_model_custom}
                
                logger.info(
                    f"Added speaker from ainame {ainame} of gender {mapped_gender} with voice {voice_model_custom}.")
                
                # if not an espiode then switch to the custom voice
                if not is_episode:
                    speaker = ainame
                    logger.info(
                        f"Text to Speech: Speaker switch from {last_speaker} to {speaker}.")
        else:
            logger.info(
                f"Text to Speech: Default settings {voice_model} at speed {voice_speed} using API {tts_api}.")

        # Find and assign voices to speakers
        story_voice_model = None
        story_gender = None
        new_speaker_count = 0
        current_speaker_count = 0

        # Regex pattern to find speaker names with different markers
        #speaker_pattern = r'^(?:\[/INST\])?<<([A-Za-z]+)>>|^(?:\[\w+\])?([A-Za-z]+):'
        # find speaker names that may have a space in them at the start of lines like . new speaker: lines or after other punctuation endings or newlines
        speaker_pattern = r'(?:(?:\[/INST\])?<<([A-Za-z0-9_\)\(\-]+)>>|^(?:\[\w+\])?([A-Za-z0-9_\)\(\-)]+):)'

        # Find speaker names in the text and derive gender from name, setup speaker map
        for line in text.split('\n'):
            speaker_match = re.search(speaker_pattern, line)
            if speaker_match:
                # Extracting speaker name from either of the capturing groups
                new_speaker = speaker_match.group(1) or speaker_match.group(2)
                new_speaker = new_speaker.strip()
                new_speaker = new_speaker.lower()

                if (new_speaker == "opening_shot" or new_speaker == "closing_shot" or new_speaker == "next_episode_summary" or new_speaker == "scene" or new_speaker == "title" or new_speaker == "episode" or new_speaker == "question" or new_speaker == "plotline" or new_speaker == "host") and new_speaker not in speaker_map:
                    speaker_map[new_speaker] = speaker_map["narrator"]

                logger.info(f"Text to Speech: Speaker #{speaker_count}/{new_speaker_count}/{current_speaker_count} found: {new_speaker}.")

                # check fuzzy match through speaker map
                speaker_found = False
                found_speaker = None
                for speaker_key in speaker_map:
                    if speaker_key in new_speaker or new_speaker in speaker_key:
                        speaker_found = True
                        found_speaker = speaker_key
                        break

                if speaker_found:
                    # check if we are really the same speaker
                    logger.info(f"Text to Speech: Speaker {speaker} #{speaker_count}/{new_speaker_count}/{current_speaker_count} fuzzy match found: {found_speaker}.")
                    #new_speaker = found_speaker
                
                if new_speaker not in speaker_map:                    
                    gender_marker = None
                    # Identify gender from text if not determined by name
                    if re.search(r'\[m\]', line):
                        gender_marker = "male"
                    elif re.search(r'\[f\]', line):
                        gender_marker = "female"
                    elif re.search(r'\[n\]', line):
                        gender_marker = "nonbinary"

                    gender_g = None
                    if gender_marker == None:
                        new_speaker_lc = new_speaker.lower()
                        if new_speaker_lc == "he-man":
                            gender_g = "male"
                        elif new_speaker_lc == "she-ra":
                            gender_g = "female"
                        elif new_speaker_lc == "skeletor":
                            gender_g = "male"
                        elif new_speaker_lc == "teela":
                            gender_g = "female"
                        elif new_speaker_lc == "shiva":
                            gender_g = "male"
                        elif new_speaker_lc == "shakti":
                            gender_g = "female"
                        else:
                            guessed_gender = d.get_gender(new_speaker.split('_')[0])  # assuming the first word is the name
                            if guessed_gender in ['male', 'mostly_male']:
                                gender_g = "male"
                            elif guessed_gender in ['female', 'mostly_female']:
                                gender_g = "female"

                    if gender_marker:
                        story_gender = gender_marker
                    elif gender_g:
                        story_gender = gender_g
                    else:
                        story_gender = last_gender
                    
                    if story_gender == "male":
                        if male_voice_index > len(male_voices):
                            male_voice_index = 0
                        voice_choice = male_voices[male_voice_index % len(male_voices)]
                        male_voice_index += 1
                    else:  # Female and nonbinary use female voices
                        if female_voice_index > len(female_voices):
                            female_voice_index = 0
                        voice_choice = female_voices[female_voice_index % len(female_voices)]
                        female_voice_index += 1

                    logger.info(f"Adding new speaker {new_speaker} {voice_choice} {story_gender}")

                    speaker = new_speaker
                    speaker_map[speaker] = {'voice': voice_choice, 'gender': story_gender}
                    story_voice_model = voice_choice
                    speaker_count += 1
                    new_speaker_count += 1
                else:
                    if new_speaker in speaker_map:
                        story_voice_model = speaker_map[new_speaker]['voice']
                        story_gender = speaker_map[new_speaker]['gender']
                    else:
                        logger.error(f"Text to Speech: ERROR Speaker {new_speaker} not found in speaker map, reassigning.")
                        story_voice_model = last_voice_model
                        story_gender = last_gender
                    speaker = new_speaker
                    if story_voice_model not in female_voices and story_voice_model not in male_voices:
                        logger.error(f"Text to Speech: ERROR Voice model {story_voice_model} not found in voice list, reassigning.")
                        if story_gender == "male":
                            if male_voice_index > len(male_voices):
                                male_voice_index = 0
                            story_voice_model = male_voices[male_voice_index]
                            male_voice_index += 1
                        else:
                            if female_voice_index > len(female_voices):
                                female_voice_index = 0
                            story_voice_model = female_voices[female_voice_index]
                            female_voice_index += 1

                current_speaker_count += 1
                logger.info(
                    f"Text to Speech: Speaker switch from {last_speaker} to #{current_speaker_count}/{speaker_count} {speaker} who is {story_gender} with voice {story_voice_model}.")
                break # only one speaker

        # show speaker map
        if new_speaker_count > 0:
            logger.info(f"Text to Speech: Current Speaker map: {json.dumps(speaker_map)}")

        # Check if we found speakers and are not in episode mode or forced to override
        if is_episode:
            if story_voice_model:
                if story_gender:
                    gender = story_gender
                logger.info(f"Text to Speech: {speaker}/{gender} Speaker #{speaker_count}/{new_speaker_count}/{current_speaker_count} found, using {story_voice_model} instead of {voice_model}.")
                voice_model = story_voice_model
            else:
                # keep the last values used
                voice_model = last_voice_model
                story_gender = last_gender
        else:
            if voice_override:
                logger.info(
                    f"Text to Speech: {speaker}/{gender} Speaker #{speaker_count}/{new_speaker_count}/{current_speaker_count} found, using {voice_model_custom} instead of {voice_model}.")
                # custom voice for single speaker mode
                voice_model = voice_model_custom
                voice_speed = voice_speed_custom
            else:
                voice_model = last_voice_model
            if gender_override:
                logger.info(f"Using custom gender {gender_custom} instead of {gender}")
                gender = gender_custom
            else:
                gender = last_gender

        # Last gender
        last_gender = gender
        last_voice_model = voice_model
        last_speaker = speaker  # Update the last speaker
        
        # clean text of end of line spaces after punctuation
        text = clean_text(text)
        text = re.sub(r'([.,!?;:])\s+', r'\1', text)

        text_flat = text.replace('\n', ' ').replace('\r', '')
        logger.debug("Text to Speech received request:\n%s" % header_message)
        logger.info(f"Text to Speech received request {speaker} {voice_model} {gender} #{segment_number}: {text_flat[:20]}...")

        # add ssml tags
        if args.ssml == 'true' and tts_api == "mimic3":
            text = f"<speak><prosody pitch=\"{args.pitch}\" range=\"{args.range}\" rate=\"{args.rate}\">" + text + f"</prosody></speak>"
            logger.info(f"Text to Speech: SSML enabled, using pitch={args.pitch}, range={args.range}, rate={args.rate}.")
            logger.debug(f"Text to Speech: SSML text:\n{text}")

        duration = 0
        try:
            audio_blob = get_tts_audio(
                tts_api,
                text,
                voice=voice_model,
                noise_scale=args.noise_scale,
                noise_w=args.noise_w,
                length_scale=voice_speed,
                ssml=args.ssml,
                audio_target=args.audio_target
            )
            if tts_api == "mimic3" or tts_api == "mms-tts":
                duration = len(audio_blob) / (22050 * 2)  # Assuming 22.5kHz 16-bit audio for duration calculation
            elif tts_api == "openai":
                duration = get_aac_duration(audio_blob)
        except Exception as e:
            logger.error(f"Exception: ERROR TTS error with API request for text: {text}")
            logger.error(e)
            continue

        if duration == 0:
            logger.error(f"Exception: ERROR TTS {tts_api} {voice_model} x{voice_speed} returned 0 duration audio blobt: {text}")
            continue

        audiobuf = io.BytesIO(audio_blob)
        audiobuf.seek(0)

        # Fill in the header
        header_message["duration"] = duration
        header_message["stream"] = "speek"

        # Send the header and the audio
        sender.send_json(header_message, zmq.SNDMORE)
        sender.send(audiobuf.getvalue())

        logger.debug(f"Text to Speech: sent audio #{segment_number}\n{header_message}")
        logger.info(f"Text to Speech: sent audio #{segment_number} of {duration} duration.")

        header_message = None
        text = ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, default=2000, required=False, help="Port for receiving text input")
    parser.add_argument("--output_port", type=int, default=6002, required=False, help="Port for sending audio output")
    parser.add_argument("--input_host", type=str, default="127.0.0.1", required=False, help="Host for receiving text input")
    parser.add_argument("--output_host", type=str, default="127.0.0.1", required=False, help="Host for sending audio output")
    parser.add_argument("--voice", type=str, default='en_US/vctk_low#p303', help="Voice parameter for TTS API")
    parser.add_argument("--noise_scale", type=str, default='0.333', help="Noise scale parameter for TTS API")
    parser.add_argument("--noise_w", type=str, default='0.333', help="Noise weight parameter for TTS API")
    parser.add_argument("--length_scale", type=str, default='1.5', help="Length scale parameter for TTS API")
    parser.add_argument("--ssml", type=str, default='false', help="SSML parameter for TTS API")
    parser.add_argument("--audio_target", type=str, default='client', help="Audio target parameter for TTS API")
    parser.add_argument("-ll", "--loglevel", type=str, default="info", help="Logging level: debug, info...")
    parser.add_argument("--rate", type=str, default="default", help="Speech rate, slow, medium, fast")
    parser.add_argument("--range", type=str, default="high", help="Speech range, low, medium, high")
    parser.add_argument("--pitch", type=str, default="high", help="Speech pitch, low, medium, high")
    parser.add_argument("--delay", type=int, default=0, help="Delay in seconds after timestamp before sending audio")
    parser.add_argument("--service", type=str, default="mimic3", help="TTS service to use. mms-tts, mimic3, openai")
    parser.add_argument("--metal", action="store_true", default=False, help="offload to metal mps GPU")
    parser.add_argument("--cuda", action="store_true", default=False, help="offload to cuda GPU")
    parser.add_argument("--gender", type=str, default="female", help="Gender default for characters without [m], [f], or [n] markers")

    args = parser.parse_args()

    LOGLEVEL = logging.INFO

    if args.loglevel == "info":
        LOGLEVEL = logging.INFO
    elif args.loglevel == "debug":
        LOGLEVEL = logging.DEBUG
    elif args.loglevel == "warning":
        LOGLEVEL = logging.WARNING
    else:
        LOGLEVEL = logging.INFO

    log_id = time.strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(filename=f"logs/ttsMimic3-{log_id}.log", level=LOGLEVEL)
    logger = logging.getLogger('tts')

    ch = logging.StreamHandler()
    ch.setLevel(LOGLEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    context = zmq.Context()
    # Set up the subscriber
    receiver = context.socket(zmq.SUB)
    logger.info(f"Setup ZMQ in {args.input_host}:{args.input_port}")
    receiver.connect(f"tcp://{args.input_host}:{args.input_port}")
    receiver.setsockopt_string(zmq.SUBSCRIBE, "")

    # Set up the publisher
    sender = context.socket(zmq.PUSH)
    logger.info(f"connected to ZMQ out {args.output_host}:{args.output_port}")
    sender.connect(f"tcp://{args.output_host}:{args.output_port}")

    model = None
    tokenizer = None
    if args.service == "mms-tts":
        model = VitsModel.from_pretrained("facebook/mms-tts-eng")
        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

        if args.metal:
            model.to("mps")
        elif args.cuda:
            model.to("cuda")
        else:
            model.to("cpu")

    openai_client = OpenAI()

    d = gender_guess.Detector(case_sensitive=False)

    main()
