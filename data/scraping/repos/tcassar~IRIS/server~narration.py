import base64

import pydub
import pydub.playback
import openai
import os
import requests
import server.image_process as imgp

openai.api_key = os.environ.get("OPENAIKEY")


class Narrator:
    """Narrates the world"""

    def __init__(self, image: imgp.ImageProcessor):
        self.messages = [
            {
                "role": "system",
                "content": "You are Iris, an AI assistant for the blind. Your hardware allows you to see the world "
                "through the user's eyes. You are trained on a large set of general images and text. "
                "Your purpose is to describe what the user is looking at and answer any questions they have. "
                "You always answer very concisely, in a single sentence.",
            }
        ]

        self.image = image

    def _generate_av_desc(
        self, usr_prompt="Please describe the scene in front of me."
    ) -> str:
        """Gives textual description of scene to gpt3.5, returns av string"""

        # set up prompts

        self.messages.append(
            {
                "role": "system",
                "content": f"The scene that the person is looking at is: {self.image.describe()}",
            },
        )

        self.messages.append({"role": "user", "content": f"{usr_prompt}"})

        # create a chat completion
        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=self.messages
        )

        response = chat_completion.choices[0].message

        self.messages.append(response)

        return response.content

    def _get_av_audio(self):
        """Get an audio.mp3 file from Google text to speech, returned as a bytestring"""

        response = requests.post(
            "https://us-central1-texttospeech.googleapis.com/v1beta1/text:synthesize",
            json=self.json,
            headers={
                "Authorization": f"Bearer {os.environ.get('GKEY')}",
                "x-goog-user-project": "sunny-truth-389311",
                "Content-Type": "application/json; charset=utf-8",
            },
        )

        return response.json()

    def narrate(self):
        """interface with which to call the narrator"""
        audio = self._get_av_audio()["audioContent"]

        # this makes me want to cry
        audio_as_mp3 = base64.decodebytes(bytes(audio, encoding="utf-8"))
        # now = datetime.now()
        #
        # with open(f"tmp/audio.mp3", "ba") as f:
        #     f.write(audio_as_mp3)
        #
        # playsound.playsound(f"tmp/audio.mp3@{now}", True)

        song = pydub.AudioSegment(audio_as_mp3, format="mp3")
        pydub.playback.play(song)


    @property
    def json(self):
        return {
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "effectsProfileId": ["small-bluetooth-speaker-class-device"],
                "pitch": 0,
                "speakingRate": 1,
            },
            "input": {"text": self._generate_av_desc()},
            "voice": {"languageCode": "en-GB", "name": "en-GB-Neural2-C"},
        }
