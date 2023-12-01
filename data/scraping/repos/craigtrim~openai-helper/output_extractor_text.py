#!/usr/bin/env python
# -*- coding: UTF-8 -*-
""" A Generic Service to Extract Unstructured Output from an OpenAI response """


from pprint import pprint
from pprint import pformat
from typing import Optional

from baseblock import Stopwatch
from baseblock import BaseObject

from openai_helper.dmo import EtlRemoveEmojis
from openai_helper.dmo import EtlReplaceCliches
from openai_helper.dmo import EtlRemoveListIndicators
from openai_helper.dmo import EtlHandleTextCompletions
from openai_helper.dmo import EtlReplaceDuplicatedInput
from openai_helper.dmo import EtlRemovePromptIndicators

LINE_BREAK = '\n'
DOUBLE_LINE_BREAK = f'{LINE_BREAK}{LINE_BREAK}'
CUSTOM_LINE_BREAK = ' CUSTOMLINEBREAK '


class OutputExtractorText(BaseObject):
    """ A Generic Service to Extract Unstructured Output from an OpenAI response """

    def __init__(self):
        """
        Created:
            17-Mar-2022
            craigtrim@gmail.com
            *   in pursuit of
                https://github.com/grafflr/graffl-core/issues/222
        Updated:
            18-Mar-2022
            craigtrim@gmail.com
            *   handle text completions
                https://github.com/grafflr/graffl-core/issues/224
        Updated:
            4-Aug-2022
            craigtrim@gmail.com
            *   migrated to 'openai-helper' in pursuit of
                https://bast-ai.atlassian.net/browse/COR-56
                https://github.com/craigtrim/openai-helper/issues/1
        Updated:
            14-Sept-2022
            craigtrim@gmail.com
            *   make text pipeline dynamic via incoming parameters
        Updated:
            16-Sept-2022
            craigtrim@gmail.com
            *   add remove-list-indicators service
                https://github.com/craigtrim/openai-helper/issues/2
        Updated:
            24-Feb-2023
            craigtrim@gmail.com
            *   add remove-emojis service
                https://github.com/craigtrim/openai-helper/issues/8
        Updated:
            1-Mar-2023
            craigtrim@gmail.com
            *   renamed from 'extract-output' in pursuit of
                https://github.com/craigtrim/openai-helper/issues/9
        """
        BaseObject.__init__(self, __name__)
        self._remove_emojis = EtlRemoveEmojis().process
        self._replace_cliched_text = EtlReplaceCliches().process
        self._remove_prompts = EtlRemovePromptIndicators().process
        self._remove_list_indicators = EtlRemoveListIndicators().process
        self._handle_text_completions = EtlHandleTextCompletions().process
        self._replace_duplicated_input = EtlReplaceDuplicatedInput().process

    @staticmethod
    def _output_text(d_result: dict) -> Optional[str]:

        if 'choices' in d_result['output']:
            choices = d_result['output']['choices']

            if len(choices):

                def get_output_text() -> str:
                    if 'text' in choices[0]:
                        return choices[0]['text'].strip()
                    raise NotImplementedError

                output_text = get_output_text()
                output_text = output_text.replace(
                    LINE_BREAK, CUSTOM_LINE_BREAK)

                while DOUBLE_LINE_BREAK in output_text:
                    output_text = output_text.replace(
                        DOUBLE_LINE_BREAK, LINE_BREAK)

                output_text = output_text.split(LINE_BREAK)[-1].strip()
                output_text = output_text.replace(
                    CUSTOM_LINE_BREAK, LINE_BREAK)

                while '  ' in output_text:
                    output_text = output_text.replace('  ', ' ')
                return output_text

    @staticmethod
    def _is_valid(d_result: dict) -> bool:
        """ Validate Incoming Result Object

        Reference:
            https://github.com/craigtrim/openai-helper/issues/4

        Args:
            d_result (dict): the OpenAI result

        Returns:
            bool: True if valid (for purposes of this ETL routine)
        """

        if not d_result:
            return False
        if 'output' not in d_result:
            return False
        if not d_result['output']:
            return False

        d_output = d_result['output']
        if 'choices' not in d_output:
            return False
        if not d_output['choices']:
            return False

        choices = d_output['choices']
        if not choices:
            return False

        for d_choice in choices:
            if 'text' not in d_choice:
                return False
            if not d_choice['text']:
                return False

        return True

    def process(self,
                input_text: str,
                d_result: dict,
                replace_duplicated_input: bool = True,
                handle_text_completions: bool = True,
                remove_prompts: bool = True,
                replace_cliched_text: bool = True,
                remove_list_indicators: bool = True,
                remove_emojis: bool = True) -> Optional[str]:
        """ Entry Point

        Args:
            input_text (str): the incoming text
            d_result (dict): the OpenAI result
            replace_duplicated_input (bool, optional): remove any duplicated text. Defaults to True.
            handle_text_completions (bool, optional): cleanse dynamic completions. Defaults to True.
            remove_prompts (bool, optional): remove any generic prompt material. Defaults to True.
            replace_cliched_text (bool, optional): removes noisy and cliched output. Defaults to True.
            remove_list_indicators (bool, optional): remove any list indicators. Defaults to True.
            remove_emojis (bool, optional): remove any emojis OpenAI might provide. Defaults to True.
                Reference:
                    https://github.com/craigtrim/openai-helper/issues/8

        Returns:
            str or None: the outgoing text
        """

        sw = Stopwatch()

        if not input_text:
            return None
        if not self._is_valid(d_result):
            if d_result:
                self.logger.error('\n'.join([
                    'Unexpected OpenAI Result Object',
                    f'\tResult: {pformat(d_result)}']))
            else:
                self.logger.error('Null OpenAI Object')
            return None

        def create_pipeline() -> list:
            text_pipeline = []

            if replace_duplicated_input:
                text_pipeline.append(self._replace_duplicated_input)

            if handle_text_completions:
                text_pipeline.append(self._handle_text_completions)

            if remove_prompts:
                text_pipeline.append(self._remove_prompts)

            if replace_cliched_text:
                text_pipeline.append(self._replace_cliched_text)

            if remove_list_indicators:
                text_pipeline.append(self._remove_list_indicators)

            if remove_emojis:
                text_pipeline.append(self._remove_emojis)

            return text_pipeline

        output_text = self._output_text(d_result)
        if not output_text or not len(output_text):
            return None

        for text_handler in create_pipeline():
            output_text = text_handler(input_text=input_text,
                                       output_text=output_text)
            if not output_text or not len(output_text):
                return None

        if output_text.startswith('"') and output_text.endswith('"'):
            output_text = output_text[1:-1]

        if self.isEnabledForDebug:
            self.logger.debug('\n'.join([
                'OpenAI Output Extraction Completed',
                f'\tTotal Time: {str(sw)}',
                f'\tInput Text: {input_text}',
                f'\tOutput Text: {output_text}']))

        return output_text
