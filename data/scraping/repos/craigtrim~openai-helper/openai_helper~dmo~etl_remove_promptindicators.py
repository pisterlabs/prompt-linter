#!/usr/bin/env python
# -*- coding: UTF-8 -*-
""" A Generic Service to Extract Unstructured Output from an OpenAI response """


from baseblock import BaseObject
from baseblock import TextMatcher


class EtlRemovePromptIndicators(BaseObject):
    """ A Generic Service to Extract Unstructured Output from an OpenAI response """

    __d_replacements = {
        'User:': '',
        'Human:': '',
        'Assistant:': '',
        'AI:': '',
        'Marv:': '',
        'Len:': '',
        "Marv's": "it's",
        "I'm an AI language model designed by OpenAI": "I'm a bot",
        'As an AI language model,': '',
        'Two-Sentence Horror Story:': '',
        'designed by OpenAI': '',
        "I'm an AI language model": "I'm a bot",
    }

    def __init__(self):
        """ Change Log

        Created:
            4-Aug-2022
            craigtrim@gmail.com
            *   https://bast-ai.atlassian.net/browse/COR-56
        Updated:
            16-Nov-2022
            craigtrim@gmail.com
            *   renamed from 'etl-remove-indicators' in pursuit of
                https://github.com/craigtrim/openai-helper/issues/2
        Updated:
            18-May-2023
            craigtrim@gmail.com
            *   move replacements into a dictionary
        """
        BaseObject.__init__(self, __name__)

    def process(self,
                input_text: str,
                output_text: str) -> str:
        """ Eliminate Annoying Situations where OpenAI responds with something like
            'Human: blah blah'
        or
            'Assistant: blah blah'

        Args:
            input_text (str): the user input text
            output_text (str): the current state of the extracted text from OpenAI

        Returns:
            str: the potentially modified output text
        """

        for k in self.__d_replacements:
            if not TextMatcher.exists(
                    value=k,
                    input_text=output_text,
                    case_sensitive=False):
                continue

            output_text = TextMatcher.replace(
                input_text=output_text,
                old_value=k,
                new_value=self.__d_replacements[k],
                case_sensitive=False,
                recursive=False)

        return output_text
