import os
from typing import List, Dict
from openai_api import OpenAIResponder
from utils import format_message
from chat_store import ChatStore

api_key = os.getenv('OPENAI_API_KEY')

class SafeguardAI:
    def __init__(self, api_key: str, model: str = 'gpt-3.5-turbo-0613', logger=None):
        self._api_key = api_key
        # initialize openai client using provided OpenAI API key
        self._model = model
        self._logger = logger
        self._responder = OpenAIResponder(api_key=api_key, model=model, logger=logger)

    def _get_response(self, messages):
        response, status, details = self._responder.get_response(messages)
        return response, status, details

    def _evaluate_factuality(self, text : str):
        prompt = f'''
        List the factual statements present in the following piece of text. Then for each such statement assess it's factuality.
        At the end write your conclusion, highlighting the inaccurate or misleading facts if there are any. Here is the text: {text}
        '
        '''

        response, status, details = self._get_response([{'role': 'system', 'content': prompt}])

        return response


    def get_evaluation(self, chat_id : str) -> str:

        conversation = ChatStore.retrieve_chat(chat_id, eval = True)

        evaluation = ''

        last_eval_idx = -1
        for idx, message in enumerate(conversation):
            if message['role'] == 'eval':
                last_eval_idx = idx

        for idx, message in enumerate(conversation):
            if idx >= last_eval_idx + 1:
                if message['role'] == 'assistant':
                    print('evaluating message: ', message['content'])
                    msg_eval = '</br>'
                    msg_eval +=  '<div class="quote"><i>' + message['content'] + '</i></div>'
                    fact_eval = self._evaluate_factuality(message['content'])
                    msg_eval += '<ul>' + format_message(fact_eval) + '</ul></br>'

                    # we store the evaluation for each message
                    ChatStore.add_message(
                        chat_id,
                        {
                            'role': 'eval', 
                            'content': msg_eval,
                            'status' : 'OK'
                        }
                    )

                    evaluation += msg_eval

        return evaluation





def _test_safeguard_ai():
    model = 'gpt-3.5-turbo-0613'

    prompt = "Q: Who starred in the 1996 blockbuster Independence Day?"
    prompt += "A: "
    
    safeguardAI = SafeguardAI(api_key=api_key, model=model)
    
    response, status, details = safeguardAI._get_response([{'role': 'system', 'content': prompt}])
    
    print('response ', response)
    print('status ', status)
    print('details ', details)


if __name__ == "__main__":
    _test_safeguard_ai()

