# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import json
import os
import time

from stage import Stage
from config import config
from audiocraft.data.audio import audio_write
from loguru import logger
from gtts import gTTS
from llm.fbmusicgen import FbMusicGen
from llm.fbaudiogen import FbAudioGen
from openai import OpenAI

class AudioStage(Stage):

    def __repr__(self) -> str:
        return 'AudioStage'

    def __str__(self) -> str:
        return self.__repr__()

    def __init__(self, _music_duration=15, _audio_duration=2):
        self.music_duration = _music_duration
        self.audio_duration = _audio_duration

    def _initialize(self):
        self.musicgen_model = FbMusicGen()
        self.musicgen_model.set_music_duration(self.music_duration)

        self.audiogen_model = FbAudioGen()
        self.audiogen_model.set_audio_duration(self.audio_duration)

    def _process(self, context):
        story_folder = os.path.join(config.server_root, config.stories_dir, context.id)
        music_filename = "music"
        tts_filename = "tts"

        for subfolder in sorted(os.listdir(story_folder)):
            subfolder_path = os.path.join(story_folder, subfolder)

            save_file_path = os.path.join(subfolder_path, '6_audio_stage.json')
            if os.path.isfile(save_file_path):
                logger.info(f"{self} step found to be already completed")
                continue
            
            # Check if the path is a directory and contains the required JSON files
            if os.path.isdir(subfolder_path):
                json_data = {}
                json_data["audio"] = {}

                if '1_analysis_stage.json' in os.listdir(subfolder_path):
                    json_path = os.path.join(subfolder_path, '1_analysis_stage.json')
                    # Read the JSON file
                    with open(json_path, 'r') as file:
                        data = json.load(file)
                        audio_data = data.get('audio', {})
                        mood_music = audio_data.get('mood', {})

                        character_data = data.get('characters', {})
                        characters = list(character_data.keys())
                        character_sound_effects = []
                        for character in characters:
                            character_sound_effects.append(character_data[character]["soundeffect"])

                        if config.UseGpuAudioGen:
                            self.musicgen_model.instantiate()
                            self.generate_music(mood_music, subfolder_path, music_filename)
                            self.musicgen_model.dispose() # as a workaround for out of memory issues, dispose resources after generating music. todo: create once to be re-used between subfolders

                            self.audiogen_model.instantiate()
                            self.generate_characters_audio(character_sound_effects, subfolder_path, characters)
                            self.audiogen_model.dispose() # as a workaround for out of memory issues, dispose resources after generating audio. todo: create once after musicgen_model has be disposed.

                            json_data["audio"]["mood"] = mood_music
                            json_data["audio"]["music_file"] = f"{music_filename}.wav"

                        json_data["audio"]["character_sound_effects"] = {}
                        if config.UseGpuAudioGen:
                            for idx, character in enumerate(characters):
                                json_data["audio"]["character_sound_effects"][character] = {}
                                json_data["audio"]["character_sound_effects"][character]["description"] = character_sound_effects[idx]
                                json_data["audio"]["character_sound_effects"][character]["path"] = f"{character}.wav"

                if 'scene.json' in os.listdir(subfolder_path):
                    json_path = os.path.join(subfolder_path, 'scene.json')
                    # Read the JSON file
                    with open(json_path, 'r') as file:
                        data = json.load(file)
                        title = data.get('title', '')
                        storycontent = data.get('content', '')
                        storycontent = self.clean_white_space(storycontent)
                        if config.UseGpuAudioGen:
                            self.generate_tts(f"{title}\n{storycontent}", subfolder_path, tts_filename)
                        else:
                            self.generate_openai_tts(f"{title}\n{storycontent}", subfolder_path, tts_filename)
                        json_data["audio"]["tts_file"] = f"{tts_filename}.mp3"
                
                if json_data:
                    with open(save_file_path, 'w') as output_file:
                        json.dump(json_data, output_file, indent=4)

        return context

    def _dispose(self):
        del self.musicgen_model
        del self.audiogen_model
        
    def generate_music(self, music_prompt, path, filename):
        wav = self.musicgen_model.generate(music_prompt)
        sample_rate = self.musicgen_model.sample_rate
        filepath = os.path.join(path, filename)
        audio_write(filepath, wav[0].cpu(), sample_rate, strategy="loudness", loudness_compressor=True)

    def generate_characters_audio(self, characters_audio_prompt, path, filenames):
        logger.info("AudioGen model about to generate for characters...")
        wav = self.audiogen_model.generate(characters_audio_prompt)
        sample_rate = self.audiogen_model.sample_rate

        for idx, one_wav in enumerate(wav):
            filepath = os.path.join(path, f"{filenames[idx]}")
            audio_write(filepath, one_wav.cpu(), sample_rate, strategy="loudness", loudness_compressor=True)

    def generate_tts(self, tts_prompt, path, filename):
        filepath = os.path.join(path, f"{filename}.mp3")
        logger.info("TTS about to generate...")

        tts = gTTS(tts_prompt, lang='en', tld='co.uk')
        tts.save(filepath)

    def generate_openai_tts(self, tts_prompt, path, filename):
        client = OpenAI(api_key=config.OpenAIApiKey)
        logger.info("TTS about to generate...")
        response = client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=tts_prompt,
        )

        filepath = os.path.join(path, f"{filename}.mp3")
        response.stream_to_file(filepath)
    
    def clean_white_space(self, str):
        return str.replace("\n", " ")
