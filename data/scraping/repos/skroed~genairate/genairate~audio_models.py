import logging
import os
import random
from functools import reduce
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

import librosa
import numpy as np
import scipy
import torch
from audiocraft.models import MusicGen
from huggingface_hub import InferenceClient
from nltk import sent_tokenize
from openai import OpenAI, _base_client
from pydub import AudioSegment
from transformers import AutoModel, AutoProcessor, AutoTokenizer, VitsModel

from genairate.utils import create_sentence_chunks

logger = logging.getLogger(__name__)

GLOBAL_SAMPLE_RATE = 22050


class AudioModel(object):
    """Abstract class for audio models."""

    def __init__(
        self,
        name: None,
        local: bool = False,
        inference_params: dict = None,
    ):
        """Initializes the AudioModel class.

        Args:
            name (None): the name of the model.
            local (bool, optional): If the model is running locally.
              Defaults to None.
            inference_params (dict, optional): Additional params to
                pass for inference. Defaults to None.
        """
        self.name = name
        self.local = local
        self.sample_rate = None

        if not inference_params:
            inference_params = {}

        self.inference_params = inference_params

    def raw_to_segment(
        self,
        data: Union[np.array, bytes, torch.tensor],
    ) -> AudioSegment:
        """Converts raw audio to an AudioSegment.

        Args:
            data (Union[np.array, bytes, torch.tensor]): The raw audio.

        Returns:
            AudioSegment: The audio as an AudioSegment.
        """
        # TODO: Cleanup this code section
        if torch.is_tensor(data):
            data = data.cpu().numpy().squeeze().astype(np.float32)
        elif isinstance(data, bytes):
            if data[:4].decode("utf8") == "fLaC":
                with TemporaryDirectory() as temp_dir:
                    file_path = Path(temp_dir) / Path("tmp_audio.flac")
                    file_path.write_bytes(data)
                    data = AudioSegment.from_file(file_path, "flac")
                    data = data.set_frame_rate(GLOBAL_SAMPLE_RATE)
                return data
            else:
                data = (
                    np.array(
                        eval(data)["generated_audio"],
                    )
                    .squeeze()
                    .astype(np.float32)
                )
        elif isinstance(data, np.ndarray):
            pass
        elif isinstance(data, _base_client.HttpxBinaryResponseContent):
            with TemporaryDirectory() as temp_dir:
                file_path = Path(temp_dir) / Path("tmp_audio.mp3")
                data.stream_to_file(file_path)
                data = AudioSegment.from_file(file_path, "mp3")
                data = data.set_frame_rate(GLOBAL_SAMPLE_RATE)
                return data
        else:
            raise ValueError(
                f"Data type {type(data)} not supported for raw_to_segment.",
            )

        if self.sample_rate != GLOBAL_SAMPLE_RATE:
            data = librosa.resample(
                data,
                orig_sr=self.sample_rate,
                target_sr=GLOBAL_SAMPLE_RATE,
            )
        with TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / Path("tmp_audio.wav")
            scipy.io.wavfile.write(file_path, GLOBAL_SAMPLE_RATE, data)
            segment = AudioSegment.from_file(file_path, "wav")
        return segment

    def get(self, prompt: str) -> AudioSegment:
        """Abstract method for getting audio from a prompt.

        Args:
            prompt (str): The prompt to generate audio from.


        Returns:
            AudioSegment: The audio generated from the prompt.
        """
        raise NotImplementedError("AudioModel is an abstract class.")


class BarkAudioModel(AudioModel):
    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        local: bool = True,
        audio_type: str = "tts",
        inference_params: dict = None,
        speaker: str = None,
    ):
        """Initializes the BarkAudioModel class.

        Args:
            model_name_or_path (str): The name or path of the model.
            device (str): The device to run the model on.
            local (bool, optional): If the bark model should run locally.
                Defaults to True.
            audio_type (str, optional): The type of the audio.
                 Defaults to "tts".
            inference_params (dict, optional): Additional inference params.
                Defaults to None.
            speaker (str, optional): The speaker to use. Defaults to None.
        """
        super().__init__(
            name="Bark",
            local=local,
            inference_params=inference_params,
        )

        self.sample_rate = 24000

        if audio_type.lower() != "tts":
            raise NotImplementedError(
                "BarkAudioModel only supports TTS at the moment.",
            )
        if not speaker:
            # Select a random speaker
            speakers = [f"v2/en_speaker_{speaker}" for speaker in range(10)]
            random.shuffle(speakers)
            self.speaker = speakers.pop()
        else:
            self.speaker = speaker

        if local:
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
            ).to(device)
            self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        else:
            self.model = InferenceClient(model=model_name_or_path)
            if ".huggingface.cloud" in model_name_or_path:
                self.custom_endpoint = True
                logger.info(
                    f"Using custom endpoint {model_name_or_path} for bark",
                )
            else:
                self.custom_endpoint = False

    def get(self, prompt: str) -> AudioSegment:
        """Gets audio from a prompt.

        Args:
            prompt (str): The prompt to generate audio from.

        Returns:
            AudioSegment: The audio generated from the prompt.
        """
        sentences = sent_tokenize(prompt)
        sentences = create_sentence_chunks(sentences)

        audio_arrays = []
        for sentence in sentences:
            logger.info(f"Generating audio for sentence: {sentence}")
            if self.local:
                audio_raw = self.model.generate(
                    **self.processor(
                        text=[sentence],
                        return_tensors="pt",
                        voice_preset=self.speaker,
                    ),
                    do_sample=False,
                )

                audio = self.raw_to_segment(audio_raw)

            else:
                if self.custom_endpoint:
                    audio_raw = self.model.post(
                        json={
                            "inputs": sentence,
                            "voice_preset": self.speaker,
                        },
                    )
                    audio = self.raw_to_segment(audio_raw)
                else:
                    logger.warning(
                        "speaker is not being used in remote setting. Use custom endpoint for this.",
                    )
                    audio_raw = self.model.post(
                        json={
                            "inputs": sentence,
                        },
                    )
                    audio = self.raw_to_segment(audio_raw)

            audio_arrays.append(audio)

        return reduce(lambda x, y: x + y, audio_arrays)


class AudiocraftAudioModel(AudioModel):
    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        local: bool = True,
        audio_type: str = "music",
        duration: int = 25,
        inference_params: dict = None,
        chunk_size: int = 30,
        audio_window: int = 20,
    ):
        """Initializes the AudiocraftAudioModel class.

        Args:
            model_name_or_path (str): The name or path of the model.
            device (str): The device to run the model on.
            local (bool, optional): If the model is running locally.
              Defaults to True.
            audio_type (str, optional): The audio type. Defaults to "music".
            duration (int, optional): How long the pieve should be.
                Defaults to 25 seconds.
            inference_params (dict, optional): additional inference params.
                Defaults to None.
            chunk_size (int, optional): The size of each chunk in seconds.
                Defaults to 30.
            audio_window (int, optional): The size of the audio window in seconds.
                Defaults to 20.
        """
        super().__init__(
            name="Audiocraft",
            local=local,
            inference_params=inference_params,
        )
        # Pass duration to inference params
        self.inference_params["duration"] = duration

        self.sample_rate = 32000
        self.chunk_size = chunk_size
        self.audio_window = audio_window

        if audio_type.lower() != "music":
            raise NotImplementedError(
                "AudiocraftAudioModel only supports music at the moment.",
            )

        if local:
            self.model = MusicGen.get_pretrained(model_name_or_path).to(device)
            self.model.set_generation_params(duration=duration)
        else:
            # Duration is 30s by default
            self.model = InferenceClient(model=model_name_or_path)
            if ".huggingface.cloud" in model_name_or_path:
                logger.info(
                    f"Using custom endpoint {model_name_or_path} for musicgen",
                )
                self.custom_endpoint = True
            else:
                self.custom_endpoint = False

    def get(self, prompt: str) -> AudioSegment:
        """Gets audio from a prompt.

        Args:
            prompt (str): The prompt to generate audio from.

        Returns:
            AudioSegment: The audio generated from the prompt.
        """
        if self.local:
            audio_raw = self.model.generate(prompt)
            audio = self.raw_to_segment(audio_raw)
            return audio
        else:
            if self.custom_endpoint:
                audio_raw = self.model.post(
                    json={
                        "inputs": prompt,
                        "generation_params": self.inference_params,
                        "chunk_size": self.chunk_size,
                        "audio_window": self.audio_window,
                    },
                )
                audio = self.raw_to_segment(audio_raw)

            else:
                audio_raw = self.model.text_to_speech(
                    prompt,
                    **self.inference_params,
                )
                audio = self.raw_to_segment(audio_raw)
            return audio


class AudioLDM2(AudioModel):
    def __init__(self, model_name_or_path: str, inference_params: dict = None):
        """Initializes the AudioLDM2 class.

        Args:
            model_name_or_path (str): The name or path of the model.
            inference_params (dict, optional): Parameters to pass for inference. Defaults to None.
        """
        super().__init__(
            name="AudioLDM2", local=False, inference_params=inference_params
        )
        self.sample_rate = 16000
        self.model = InferenceClient(model=model_name_or_path)
        if ".huggingface.cloud" in model_name_or_path:
            logger.info(
                f"Using custom endpoint {model_name_or_path} for audildm2",
            )
            self.custom_endpoint = True
        else:
            self.custom_endpoint = False

    def get(self, prompt: str) -> AudioSegment:
        """Gets audio from a prompt.

        Args:
            prompt (str): The prompt to generate audio from.

        Returns:
            AudioSegment: The audio generated from the prompt.
        """

        if self.custom_endpoint:
            audio_raw = self.model.post(
                json={
                    "inputs": prompt,
                    "duration": self.inference_params.get("duration", 30),
                    "negative_prompt": self.inference_params.get(
                        "negative_prompt", "Low quality, average quality."
                    ),
                },
            )
            audio = self.raw_to_segment(audio_raw)

        else:
            audio_raw = self.model.text_to_speech(
                prompt,
                **self.inference_params,
            )
            audio = self.raw_to_segment(audio_raw)
        return audio


class VitsAudioModel(AudioModel):
    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        local: bool = True,
        audio_type: str = "tts",
        inference_params: dict = None,
    ):
        """Initializes the VitsAudioModel class.

        Args:
            model_name_or_path (str): The name or path of the model.
            device (str): The device to run the model on.
            local (bool, optional): If its running locally.
                Defaults to True.
            audio_type (str, optional): Which audio type. Defaults to "tts".
            inference_params (dict, optional): additional inference params.
                Defaults to None.
        """
        super().__init__(
            name="Vits",
            local=local,
            inference_params=inference_params,
        )

        if audio_type.lower() != "tts":
            raise NotImplementedError(
                "VitsAudioModel only supports TTS at the moment.",
            )

        if local:
            self.model = VitsModel.from_pretrained(
                model_name_or_path,
            ).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        else:
            self.model = InferenceClient(model=model_name_or_path)

    def get(self, prompt: str) -> AudioSegment:
        """Gets audio from a prompt.

        Args:
            prompt (str): The prompt to generate audio from.

        Returns:
            AudioSegment: The audio generated from the prompt.
        """
        sentences = sent_tokenize(prompt)

        audio_arrays = []
        for sentence in sentences:
            logger.info(f"Generating audio for sentence: {sentence}")
            if self.local:
                inputs = self.tokenizer(sentence, return_tensors="pt")
                with torch.no_grad():
                    audio_raw = self.model(
                        **inputs,
                    ).waveform
                audio = self.raw_to_segment(audio_raw)

            else:
                audio_raw = self.model.post(
                    json={
                        "inputs": sentence,
                    },
                )
                audio = self.raw_to_segment(audio_raw)
            audio_arrays.append(audio)

        return reduce(lambda x, y: x + y, audio_arrays)


class OpenAiAudioModel(AudioModel):
    def __init__(self, model_name_or_path: str, voice_preset: str = None):
        """Initializes the OpenAiAudioModel class.

        Args:
            model_name_or_path (str): The name or path of the model.
            voice_preset (str, optional): The voice preset. Defaults to None.
        """
        super().__init__(
            name="OpenAi",
            local=False,
        )
        self.inference_client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
        )
        self.model_name_or_path = model_name_or_path
        if not voice_preset:
            voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
            self.voice_preset = random.choice(voices)
        else:
            self.voice_preset = voice_preset

    def get(self, prompt: str) -> AudioSegment:
        """Gets audio from a prompt.

        Args:
            prompt (str): The prompt to generate audio from.

        Returns:
            AudioSegment: The audio generated from the prompt.
        """
        response = self.inference_client.audio.speech.create(
            model=self.model_name_or_path,
            voice=self.voice_preset,
            input=prompt,
        )
        segment = self.raw_to_segment(response)
        return segment


def get_audio_model(config: dict) -> AudioModel:
    """Helper function to get an audio model from the config. Currently
    supports Bark, Audiocraft, and Vits.

    Args:
        config (dict): The config for the audio model.

    Returns:
        AudioModel: The initialized audio model.
    """
    if config["name"].lower() == "bark":
        return BarkAudioModel(
            model_name_or_path=config["model_name_or_path"],
            device=config.get("device", "cpu"),
            local=config["local"],
            audio_type=config["audio_type"],
            speaker=config.get("speaker", None),
        )
    elif config["name"].lower() == "audiocraft":
        return AudiocraftAudioModel(
            model_name_or_path=config["model_name_or_path"],
            device=config.get("device", "cpu"),
            local=config["local"],
            audio_type=config["audio_type"],
            duration=config.get("duration", None),
            inference_params=config.get("generation_params", {}),
            chunk_size=config.get("chunk_size", 30),
            audio_window=config.get("audio_window", 20),
        )
    elif config["name"].lower() == "vits":
        return VitsAudioModel(
            model_name_or_path=config["model_name_or_path"],
            device=config.get("device", "cpu"),
            local=config["local"],
            audio_type=config["audio_type"],
        )
    elif config["name"].lower() == "openai":
        return OpenAiAudioModel(
            model_name_or_path=config["model_name_or_path"],
            voice_preset=config.get("voice_preset", None),
        )
    elif config["name"].lower() == "audioldm2":
        return AudioLDM2(
            model_name_or_path=config["model_name_or_path"],
            inference_params=config.get("inference_params", None),
        )
    else:
        raise ValueError(f"Model {config['name']} not supported.")
