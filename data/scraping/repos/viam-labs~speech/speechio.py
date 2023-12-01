from typing import ClassVar, Mapping
from enum import Enum
import time
import os
import re
import json
import asyncio
import hashlib
from typing_extensions import Self

from viam.module.types import Reconfigurable
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.types import Model
from viam import logging

import pygame
from pygame import mixer
import elevenlabs as eleven
import pygame._sdl2 as sdl2
from gtts import gTTS
import openai
import speech_recognition as sr

from .api import SpeechService
LOGGER = logging.getLogger(__name__)
CACHEDIR = "/tmp/cache"

mixer.init(buffer=1024)
rec_state = {}

class SpeechProvider(Enum):
    google = "google"
    elevenlabs = "elevenlabs"

class CompletionProvider(Enum):
    openaigpt35turbo = "openai"

class RecState():
    listen_closer = None
    mic = None
    rec = None

rec_state = RecState()

class SpeechIOService(SpeechService, Reconfigurable):
    """This is the specific implementation of a ``SpeechService`` (defined in api.py)

    It inherits from SpeechService, as well as conforms to the ``Reconfigurable`` protocol, which signifies that this component can be
    reconfigured. It also specifies a function ``SpeechIOService.new``, which conforms to the ``resource.types.ResourceCreator`` type,
    which is required for all models.
    """

    MODEL: ClassVar[Model] = Model.from_string("viam-labs:speech:speechio")
    speech_provider: SpeechProvider
    speech_provider_key: str
    speech_voice: str
    completion_provider: CompletionProvider
    completion_model: str
    completion_provider_org: str
    completion_provider_key: str
    completion_persona: str
    listen: bool
    listen_trigger_say: str
    listen_trigger_completion: str
    listen_trigger_command: str
    listen_command_buffer_length: int
    mic_device_name: str
    command_list: list
    trigger_active: bool
    active_trigger_type: str
    disable_mic: bool

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        speechio = cls(config.name)
        speechio = speechio.reconfigure(config, dependencies)

        LOGGER.debug(json.dumps(speechio.__dict__))
        return speechio

    async def say(self, text: str, blocking: bool, cache_only: bool = False) -> str:
        if str == "":
            raise ValueError("No text provided")

        LOGGER.info("Generating audio...")
        if not os.path.isdir(CACHEDIR):
            os.mkdir(CACHEDIR)

        file = CACHEDIR + '/' + self.speech_provider + self.speech_voice + self.completion_persona + hashlib.md5(text.encode()).hexdigest() + ".mp3"
        try:
            if not os.path.isfile(file): # read from cache if it exists
                if (self.speech_provider == 'elevenlabs'):
                    audio = eleven.generate(text=text, voice=self.speech_voice)
                    eleven.save(audio=audio, filename=file)
                else:
                    sp = gTTS(text=text, lang='en', slow=False)
                    sp.save(file)

            if not cache_only:
                mixer.music.load(file) 
                LOGGER.info("Playing audio...")
                mixer.music.play() # Play it

                if blocking == True:
                    while mixer.music.get_busy():
                        pygame.time.Clock().tick()
            
                LOGGER.info("Played audio...")
        except RuntimeError:
            raise ValueError("say() speech failure")

        return text

    async def listen_trigger(self, type: str) -> str:
        if type == '':
            raise ValueError("No trigger type provided")
        if type in ['command', 'completion', 'say']:
            self.active_trigger_type = type
            self.trigger_active = True
            if self.listen:
                # close and re-open listener so any in-progress speech is not captured
                rec_state.listen_closer(True)
            rec_state.listen_closer = rec_state.rec.listen_in_background(source=rec_state.mic, phrase_time_limit=self.listen_phrase_time_limit, callback=self.listen_callback)
        else:
            raise ValueError("Invalid trigger type provided")
        
        return "OK"

    async def is_speaking(self) -> bool:        
        return mixer.music.get_busy()
      
    async def completion(self, text: str, blocking: bool, cache_only: bool = False) -> str:
        if text == "":
            raise ValueError("No text provided")
        if self.completion_provider_org == '' or self.completion_provider_key == '':
            raise ValueError("completion_provider_org or completion_provider_key missing")
        
        completion = ""
        file = CACHEDIR + '/' + self.speech_provider + self.completion_persona + hashlib.md5(text.encode()).hexdigest() + ".txt"
        if not cache_only and (self.cache_ahead_completions == True):
            LOGGER.info("Will try to read completion from cache")
            if os.path.isfile(file):
                LOGGER.info("Cache file exists")
                with open(file) as f:
                    completion = f.read()
                LOGGER.info(completion)

            # now cache next one
            asyncio.ensure_future(self.completion(text, blocking, True))

        if completion == "":
            LOGGER.info("Getting completion...")
            if self.completion_persona != '':
                text = "As " + self.completion_persona + " respond to '" + text + "'"
            completion = await openai.ChatCompletion.acreate(model=self.completion_model, max_tokens=1024, messages=[{"role": "user", "content": text}])
            completion = completion.choices[0].message.content
            completion = re.sub('[^0-9a-zA-Z.!?,:\'/ ]+', '', completion).lower()
            completion = completion.replace("as an ai language model", "")
            LOGGER.info("Got completion...")

        if cache_only:
            with open(file, 'w') as f:
                f.write(completion)
            asyncio.ensure_future(self.say(completion, blocking, True))
        else:
            await self.say(completion, blocking)
        return completion
        
    async def get_commands(self, number: int) -> list:
        LOGGER.info("will get " + str(number) + " commands from command list")
        to_return = self.command_list[0:number]
        LOGGER.debug("to return from command_list: " + str(to_return))
        del self.command_list[0:number]
        return to_return

    def listen_callback(self, recognizer, audio):
        try:
            # for testing purposes, we're just using the default API key
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            # instead of `r.recognize_google(audio)`
            transcript = recognizer.recognize_google(audio,show_all=True)
            if type(transcript) is dict and transcript.get("alternative"):
                heard = transcript["alternative"][0]["transcript"]
                LOGGER.debug("speechio heard " + heard)
                if (self.listen and re.search(".*" + self.listen_trigger_say, heard))or (self.trigger_active and self.active_trigger_type == 'say'):
                    self.trigger_active = False
                    to_say = re.sub(".*" + self.listen_trigger_say + "\s+",  '', heard)
                    asyncio.run(self.say(to_say))
                elif (self.listen and re.search(".*" + self.listen_trigger_completion, heard)) or (self.trigger_active and self.active_trigger_type == 'completion'):
                    self.trigger_active = False
                    to_say = re.sub(".*" + self.listen_trigger_completion + "\s+",  '', heard)
                    asyncio.run(self.completion(to_say))
                elif (self.listen and re.search(".*" + self.listen_trigger_command, heard)) or (self.trigger_active and self.active_trigger_type == 'command'):
                    self.trigger_active = False
                    command = re.sub(".*" + self.listen_trigger_command + "\s+",  '', heard)
                    self.command_list.insert(0, command)
                    LOGGER.debug("added to command_list: '" + command + "'")
                    del self.command_list[self.listen_command_buffer_length:]
                if not self.listen:
                    # stop listening if not in background listening mode
                    LOGGER.debug("will close background listener")
                    rec_state.listen_closer()
        except sr.UnknownValueError:
            LOGGER.warn("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            LOGGER.warn("Could not request results from Google Speech Recognition service; {0}".format(e))

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        self.speech_provider = config.attributes.fields["speech_provider"].string_value or 'google'
        self.speech_provider_key = config.attributes.fields["speech_provider_key"].string_value or ''
        self.speech_voice = config.attributes.fields["speech_voice"].string_value or 'Josh'
        self.completion_provider = config.attributes.fields["completion_provider"].string_value or 'openai'
        self.completion_model = config.attributes.fields["completion_model"].string_value or 'gpt-4'
        self.completion_provider_org = config.attributes.fields["completion_provider_org"].string_value or ''
        self.completion_provider_key = config.attributes.fields["completion_provider_key"].string_value or ''
        self.completion_persona = config.attributes.fields["completion_persona"].string_value or ''
        self.listen = config.attributes.fields["listen"].bool_value or False
        self.listen_phrase_time_limit = config.attributes.fields["listen_phrase_time_limit"].number_value or None
        self.mic_device_name = config.attributes.fields["mic_device_name"].string_value or ""
        self.listen_trigger_say = config.attributes.fields["listen_trigger_say"].string_value or "robot say"
        self.listen_trigger_completion = config.attributes.fields["listen_trigger_completion"].string_value or "hey robot"
        self.listen_trigger_command = config.attributes.fields["listen_trigger_command"].string_value or "robot can you"
        self.listen_command_buffer_length = config.attributes.fields["listen_command_buffer_length"].number_value or 10
        self.cache_ahead_completions = config.attributes.fields["cache_ahead_completions"].bool_value or False
        self.disable_mic = config.attributes.fields["disable_mic"].bool_value or False
        self.command_list = []
        self.trigger_active = False
        self.active_trigger_type = ''

        if self.speech_provider == 'elevenlabs' and self.speech_provider_key != '':
            eleven.set_api_key(self.speech_provider_key)
        else:
            self.speech_provider = 'google'
        
        if self.completion_provider_org:
            openai.organization = self.completion_provider_org
        if self.completion_provider_key:
            openai.api_key = self.completion_provider_key

        if not self.disable_mic:
            # set up speech recognition
            if rec_state.listen_closer != None:
                rec_state.listen_closer(True)
            rec_state.rec = sr.Recognizer()
            rec_state.rec.dynamic_energy_threshold = True

            mics = sr.Microphone.list_microphone_names()
            LOGGER.info(mics)
            if self.mic_device_name != "":
                rec_state.mic = sr.Microphone(mics.index(self.mic_device_name))
            else:
                rec_state.mic = sr.Microphone()
            
            with rec_state.mic as source:
                rec_state.rec.adjust_for_ambient_noise(source, 2)

            # set up background listening if desired
            if self.listen == True:
                LOGGER.debug("Will listen in background")
                rec_state.listen_closer = rec_state.rec.listen_in_background(source=rec_state.mic, phrase_time_limit=self.listen_phrase_time_limit, callback=self.listen_callback)

        return self