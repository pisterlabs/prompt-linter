from typing import Union

import openai
from openai import AsyncOpenAI, OpenAI

from greptimeai.trackee import Trackee

from . import OpenaiTrackees


class _SpeechTrackees:
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        audio_speech_create = Trackee(
            obj=client.audio.speech if client else openai.audio.speech,
            method_name="create",
            span_name="audio.speech.create",
        )

        audio_raw_speech_create = Trackee(
            obj=client.audio.with_raw_response.speech
            if client
            else openai.audio.with_raw_response.speech,
            method_name="create",
            span_name="audio.with_raw_response.speech.create",
        )

        audio_speech_raw_create = Trackee(
            obj=client.audio.speech.with_raw_response
            if client
            else openai.audio.speech.with_raw_response,
            method_name="create",
            span_name="audio.speech.with_raw_response.create",
        )

        self.trackees = [
            audio_speech_create,
            audio_raw_speech_create,
            audio_speech_raw_create,
        ]

        if client:
            raw_audio_speech_create = Trackee(
                obj=client.with_raw_response.audio.speech,
                method_name="create",
                span_name="with_raw_response.audio.speech.create",
            )
            self.trackees.append(raw_audio_speech_create)


class _TranscriptionTrackees:
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        audio_transcriptions_create = Trackee(
            obj=client.audio.transcriptions if client else openai.audio.transcriptions,
            method_name="create",
            span_name="audio.transcriptions.create",
        )

        audio_raw_transcriptions_create = Trackee(
            obj=client.audio.with_raw_response.transcriptions
            if client
            else openai.audio.with_raw_response.transcriptions,
            method_name="create",
            span_name="audio.with_raw_response.transcriptions.create",
        )

        audio_transcriptions_raw_create = Trackee(
            obj=client.audio.transcriptions.with_raw_response
            if client
            else openai.audio.transcriptions.with_raw_response,
            method_name="create",
            span_name="audio.transcriptions.with_raw_response.create",
        )

        self.trackees = [
            audio_transcriptions_create,
            audio_raw_transcriptions_create,
            audio_transcriptions_raw_create,
        ]

        if client:
            raw_audio_transcriptions_create = Trackee(
                obj=client.with_raw_response.audio.transcriptions,
                method_name="create",
                span_name="with_raw_response.audio.transcriptions.create",
            )
            self.trackees.append(raw_audio_transcriptions_create)


class _TranslationTrackees:
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        audio_translations_create = Trackee(
            obj=client.audio.translations if client else openai.audio.translations,
            method_name="create",
            span_name="audio.translations.create",
        )

        audio_raw_translations_create = Trackee(
            obj=client.audio.with_raw_response.translations
            if client
            else openai.audio.with_raw_response.translations,
            method_name="create",
            span_name="audio.with_raw_response.translations.create",
        )

        audio_translations_raw_create = Trackee(
            obj=client.audio.translations.with_raw_response
            if client
            else openai.audio.translations.with_raw_response,
            method_name="create",
            span_name="audio.translations.with_raw_response.create",
        )

        self.trackees = [
            audio_translations_create,
            audio_raw_translations_create,
            audio_translations_raw_create,
        ]

        if client:
            raw_audio_translations_create = Trackee(
                obj=client.with_raw_response.audio.translations,
                method_name="create",
                span_name="with_raw_response.audio.translations.create",
            )
            self.trackees.append(raw_audio_translations_create)


class AudioTrackees(OpenaiTrackees):
    def __init__(self, client: Union[OpenAI, AsyncOpenAI, None] = None):
        speech_trackees = _SpeechTrackees(client)
        transcription_trackees = _TranscriptionTrackees(client)
        translation_trackees = _TranslationTrackees(client)

        trackees = (
            speech_trackees.trackees
            + transcription_trackees.trackees
            + translation_trackees.trackees
        )

        super().__init__(trackees=trackees, client=client)
