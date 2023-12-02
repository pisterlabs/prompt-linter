
import os
import logging
import json
from typing import Tuple
import time

from dotenv import load_dotenv

from partygpt.comms import OpenaiChat
from partygpt.comms.various import read_yaml
from partygpt.humanly.personas import Persona


logger = logging.getLogger(__name__)
load_dotenv()


class Entity:
    def __init__(
            self,
            path_settings: str,
            conversation_record_folder_path: str='') -> None:

        self._settings = read_yaml(path_settings)
        self._model = self._settings['model']['id']
        self._openai_chat = OpenaiChat(organisation=os.getenv('OPENAI_ORG'),
                                       api_key=os.getenv('OPENAI_API_KEY'))

        self._functions = {}
        self._functions_doc = []

        self._messages_history = []
        self._conversation_record_folder_path = conversation_record_folder_path

    def get_accumulated_costs(self, verbose=True):
        total_t = self._openai_chat.get_total_tokens()
        if verbose:
            print(f'Tokens accumulated:', total_t['sum'])
        # TODO: price per 1K tokens
        return total_t['sum']

    def _clear_messages_history(self) -> None:
        self._messages_history = []

    def save_messages_history(self) -> None:
        """Needs to be called before reset.
        """
        if self._conversation_record_folder_path and self._messages_history:  # only record if path defined
            try:
                os.makedirs(self._conversation_record_folder_path, exist_ok=True)
                file_name = os.path.join(self._conversation_record_folder_path,
                                        str(self._current_conversation_start) + '.convrec')
                with open(file_name, 'x', encoding='utf-8') as f:
                    for el in self._messages_history:
                        role_ = el['role']
                        content_ = el['content']
                        if role_ in ['system']:
                            continue

                        if role_ == 'user':
                            actor_ = 'User'
                        elif role_ == 'assistant':
                            actor_ = 'AI'
                        else:
                            actor_ = 'undefined'
                        f.write(actor_ + ':\n' + content_ + '\n\n')
                logger.info(f'Wrote conversation to {file_name}.')
            except Exception as e:
                logger.error(f'Trouble writing conversation history to file: {type(e).__name__}: {e}')
        else:
            logger.error('Conversation history not saved. Check if conversation record folder path '
                         'is defined and message history is not cleared')


    def set_functions(
            self,
            functions: list,
            functions_doc: list) -> None:
        if not functions or not functions_doc:
            raise ValueError('Empty list(s) provided')

        functions_names = [el.__name__ for el in functions]
        functions_doc_names = [el['name'] for el in functions_doc]
        if not (set(functions_names) == set(functions_doc_names)):
            raise ValueError('Mismatch between passed functions and function documentations. '
                             f'functions: {functions_names}, function documentations: '
                             f'{functions_doc_names}')

        self._functions = dict(zip(functions_names, functions))
        self._functions_doc = functions_doc

    def reset(self):
        self._clear_messages_history()
        self._current_conversation_start = int(time.time())

    def set_conversation_record_folder_path(self, new_path):
        self._conversation_record_folder_path = new_path
        if new_path != '':
            logger.info(f'Chat will be recorded to {new_path}')
        else:
            logger.info('Chat will not be recorded')    

    def communicate(
            self,
            messages: list,
            wait_s: float=0.5,
            max_tokens: int=150,
            # temperature: float=1.0
            ) -> Tuple[str, dict]:
        response = self._openai_chat.request_completion(
            messages=messages,
            model=self._model,
            wait_s=wait_s,
            max_tokens=max_tokens,
            # temperature=temperature,
            functions=self._functions_doc)
        return response

    def count_messages(self, role: str='none') -> int:
        if role not in ['system', 'user', 'assistant', 'none']:
            raise ValueError(f'Unexpected role "{role}".')
        if len(self._messages_history) == 0:
            return 0

        if role == 'none':
            return len(self._messages_history)
        else:
            count = 0
            for el in self._messages_history:
                if el['role'] == role:
                    count += 1
            return count


class AiGuest(Entity):
    def __init__(
            self,
            path_settings: str,
            persona: Persona=None,
            functions: list=[],
            conversation_record_folder_path: str='') -> None:
        super().__init__(path_settings=path_settings,
                         conversation_record_folder_path=conversation_record_folder_path)
        self._guest_settings = self._settings['guest']

        self.reset(persona=persona)

        self.set_functions(
            functions=functions,
            functions_doc=self._guest_settings['function_documentations'])

    def reset(
            self,
            persona: Persona=None) -> None:
        super().reset()

        if persona is not None:
            self._persona = persona
        else:
            self._persona = Persona(sampling_space=self._guest_settings['persona_space'])

        self._system_content = \
            self._guest_settings['system_content_intro'] + ' ' + \
            self._persona.characterize() + ' ' + \
            self._guest_settings['system_content_outro']

    def communicate(self, user_input: str) -> str:
        if not self._messages_history:  # initial call (and after clearing messages history)
            messages = self._openai_chat.assemble_request_messages(
                system_content=self._system_content,
                user_prompt=user_input)
        else:  # augment history
            messages = [{'role': 'user', 'content': user_input}]

        self._messages_history += messages

        response_msg, response_fct = super().communicate(
            messages=self._messages_history)

        if response_fct:
            fct_name = response_fct['name']
            fct_args = json.loads(response_fct['arguments'])

            if fct_name == 'end_communication':
                self._messages_history += [{'role': 'assistant', 'content': fct_args['goodbye']}]
            else:
                pass  # TODO: append to self._messages_history in other cases?

            logger.info(f'Assistant requested calling "{fct_name}" with args {fct_args}.')
            try:
                fct_to_call = self._functions[fct_name]
            except KeyError:
                logger.error(f'Function not defined.')
            try:
                fct_to_call(**fct_args)  # TODO: process return values if defined?
            except TypeError:
                logger.error(f'Function received unexpected keyword arguments.')

        if response_msg:
            self._messages_history += [{'role': 'assistant', 'content': response_msg}]

        logger.debug(f'Current _messages_history: {self._messages_history}')

        return response_msg
