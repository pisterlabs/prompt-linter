#!/usr/bin/env python
# -*- coding: UTF-8 -*-
""" A Generic Service to Remove Emoji Output from an OpenAI response """


from baseblock import BaseObject


class EtlRemoveEmojis(BaseObject):
    """ A Generic Service to Remove Emoji Output from an OpenAI response """

    def __init__(self):
        """ Change Log

        Created:
            25-Feb-2023
            craigtrim@gmail.com
            *   https://github.com/craigtrim/openai-helper/issues/8
        """
        BaseObject.__init__(self, __name__)

    def process(self,
                input_text: str,
                output_text: str) -> str:
        """ Entry Point

        Args:
            input_text (str): the user input text
            output_text (str): the current state of the extracted text from OpenAI

        Returns:
            str: the potentially modified output text
        """

        if output_text.count(':') < 2:
            return output_text

        def is_emoji(token: str) -> bool:
            if not token.startswith(':'):
                return False
            if not token.endswith(':'):
                return False
            return True

        output_text = ' '.join([
            x for x in output_text.split()
            if not is_emoji(x)
        ]).strip()

        return output_text
