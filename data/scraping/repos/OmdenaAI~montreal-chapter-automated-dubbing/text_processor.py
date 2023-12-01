import openai
from typing import Dict, List, Union
from translate import Translator

from .config import OPENAI_API_KEY, OPENAI_API_ORGANIZATION


class TextProcessor:
    def translate_text_segments(self, text_dict: dict, target_language: str) -> Dict[str, List[Union[str, int]]]:
        """
        Perform text translation of all text segments.

        Parameters
        ----------
        text_dict: dict
            The dictionary of transcribed, timestamped, text segments. Dictionary elements:
                - segments: list - list of dictionaries with the following elements:
                    - id: int - segment sequence number
                    - start: int - start time of phrase within the audio, in milliseconds
                    - end: int - end time of phrase within the audio, in milliseconds
                    - speaker_gender: str - auto-detected gender of speaker ('Female' / 'Male' / '')
                    - text: str - phrase text
                    - audio: AudioSegment - corresponding generated audio (empty, filled later)
                - original_language: str - detected language in original audio (code in ISO_639-1 format, as in .config)
                - original_text: str - transcribed text in full

        target_language: str
            The target language for the translation (language code)

        Returns
        -------
        dict
            Same format as input, only "text" in each segment is replaced by the translated text

        Example:

        translated_text_dict = {
            'segments': [
              {'id': 0,
               'start': 0.0,
               'end': 9.82,
               'speaker_gender': 'Female',
               'text': 'What is the worst insect on the planet?'},
               'audio': None},
              {'id': 1,
               'start': 15.48,
               'end': 17.16,
               'speaker_gender': 'Male',
               'text': 'The mosquito.',
               'audio': None}
               ],
            'original_language': 'es',
            'original_text': 'What is the worst insect on the planet? The mosquito.'
        }
        """

        # iterate through the original text_segments to create a similar list of dicts but with text translated
        translated_list = []
        for segment in text_dict['segments']:
            new_text = self.translate_text(segment['text'], target_language, text_dict['original_language'])
            new_dict = segment.copy()
            new_dict['text'] = new_text
            # new_dict["language"] = target_language
            translated_list.append(new_dict)

        translated_text_dict = text_dict
        translated_text_dict['segments'] = translated_list

        return translated_text_dict

    @staticmethod
    def translate_text(text: str, to_language: str, from_language: str = None) -> str:
        """
        Translate a single text string. If not provided in ```from_language```, original language is auto-detected.

        Parameters
        ----------
        text: str
            Text to translate
        to_language: str
            The target language for the translation. 2-letter language code in ISO_639-1 format, as defined in .config
        from_language: str
            The original language in text. 2-letter language code in ISO_639-1 format, as defined in .config

        Returns
        -------
        The translated string.
        """

        translator = Translator(to_lang=to_language, from_lang=from_language)
        translation = translator.translate(text)

        return translation

    def shorten_text_segments(self, text_dict: dict) -> Dict[str, List[Union[str, int]]]:
        """
        Shorten the translated text segments of the corresponding audio that exceeds the segment slot duration.

        Parameters
        ----------
        text_dict: dict
            The dictionary of text and audio segments.

        Returns
        -------
        dict
            Same format as input, only "text" in each segment is replaced by the shortened text

        Example:

        translated_text_dict = {
            'segments': [
              {'id': 0,
               'start': 0.0,
               'end': 9.82,
               'speaker_gender': 'Female',
               'text': 'What is the worst insect on the planet?',
               'audio': <pydub.audio_segment.AudioSegment>},
              {'id': 1,
               'start': 15.48,
               'end': 17.16,
               'speaker_gender': 'Male',
               'text': 'The mosquito.',
               'audio': <pydub.audio_segment.AudioSegment>}
               ],
            'original_language': 'es',
            'original_text': 'What is the worst insect on the planet? The mosquito.'
        }
        """

        # iterate through the text segments and shorten those, and only those, that don't fit the available time slot.
        translated_list = []
        for segment in text_dict['segments']:
            new_segment = segment.copy()

            # current audio segment duration, in seconds
            audio_duration = len(segment['audio']) / 1000

            # available segment slot duration, in seconds
            slot_duration = segment['end'] - segment['start']

            # in case audio is longer than the available slot, shorten the text
            if audio_duration > slot_duration:
                new_segment['text'] = self.shorten_text(segment['text'])

            # new_dict["language"] = target_language
            translated_list.append(new_segment)

        translated_text_dict = text_dict
        translated_text_dict['segments'] = translated_list

        return translated_text_dict

    @staticmethod
    def shorten_text(text):
        """
        Attempts to shorten a text string by summarization.
        """

        openai.api_key = OPENAI_API_KEY
        openai.organization = OPENAI_API_ORGANIZATION

        new_text = text

        if OPENAI_API_KEY:
            prompt = f"""
            Summarize the text delimited by triple backticks using fewer tokens.
            ```{text}```
            """
            messages = [{"role": "user", "content": prompt}]

            try:
                # Use GPT to shorten the text
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0,
                )
                new_text = response.choices[0].message["content"]
            except:
                pass

            # In case the newly generated text is actually longer than the original (in terms of number of words),
            # keep the original text
            if len(new_text.split()) > len(text.split()):
                new_text = text

        return new_text
