#!/usr/bin/env python
# -*- coding: UTF-8 -*-
""" Replace Cliched Responses, which just add noise to the output """


from baseblock import BaseObject


class EtlReplaceCliches(BaseObject):
    """ A Generic Service to Extract Unstructured Output from an OpenAI response """

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
        """ Replace Cliched Responses, which just add noise to the output

        Args:
            input_text (str): the user input text
            output_text (str): the current state of the extracted text from OpenAI

        Returns:
            str: the potentially modified output text
        """

        long_texts = [
            "and that's where Loqi comes in.",
            "If you're looking for a chatbot that will give you sassy responses to your questions",
            'look no further than Loqi',
            "He may not be the most helpful chatbot out there, but he's definitely the funniest",
            'Loqi is a chatbot that reluctantly answers questions in a mocking tone'
            'is a chatbot that responds to questions with',
            'is a chatbot that reluctantly answers questions',
        ]

        for long_text in long_texts:
            if long_text in output_text:
                output_text = output_text.replace(long_text, '')

        return output_text
