#!/usr/bin/env python
# -*- coding: UTF-8 -*-
""" Handle Situations where OpenAI tries to complete a User Sentence """


from baseblock import BaseObject


class EtlHandleTextCompletions(BaseObject):
    """ Handle Situations where OpenAI tries to complete a User Sentence """

    def __init__(self):
        """ Change Log

        Created:
            4-Aug-2022
            craigtrim@gmail.com
            *   https://bast-ai.atlassian.net/browse/COR-56
        """
        BaseObject.__init__(self, __name__)

    def process(self,
                input_text: str,
                output_text: str) -> str:
        """ Handle Situations where OpenAI tries to complete a User Sentence
        These are not imperative to resolve, but resolution frequently results in cleaner output

        Args:
            input_text (str): the user input text
            output_text (str): the current state of the extracted text from OpenAI

        Returns:
            str: the potentially modified output text
        """

        # this represents openAI trying to complete a user sentence
        # openAI will generally do this if the user does not terminate their input with punctuation like .!?
        # graffl now adds ending punctuation where none exists, so this pattern rarely takes place ...
        if output_text.startswith(' ') and '\n\n' in output_text:
            response = output_text.split('\n\n')[-1].strip()
            if response and len(response):
                return response

        # this is more common and seems to represent another form of text completion
        # an example is "0\n\nI'm not sure what you're asking"
        # the text prior to the response tends to be brief
        if '\n\n' in output_text:
            lines = output_text.split('\n\n')
            lines = [x.strip() for x in lines if x]
            lines = [x for x in lines if len(x) > 5]
            output_text = ' '.join(lines)
            while '  ' in output_text:
                output_text = output_text.replace('  ', ' ')

        if output_text.startswith('"') and output_text.endswith('"'):
            output_text = output_text[1:len(output_text) - 1]

        # Would I not agree that this is the most important issue of our time? [Sarcastic tone]
        if '[' in output_text and ']' in output_text:
            x = output_text.index('[')
            y = output_text.index(']')
            output_text = f'{output_text[:x]}{output_text[y + 1:]}'.strip()

        return output_text
