import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'openai_utils'))

import json
import logging
from dispatchers import utils

from openai_utils.openai_langchain_sample import OpenAILangchainBot


logger = utils.get_logger(__name__)
logger.setLevel(logging.DEBUG)
CHAT_HISTORY='chat_history'
initial_history = {CHAT_HISTORY: f'AI: Hi there! How Can I help you?\nHuman: ',}


class LexV2SMOpenAILangchainDispatcher():

    def __init__(self, intent_request):
        # See lex bot input format to lambda https://docs.aws.amazon.com/lex/latest/dg/lambda-input-response-format.html
        self.intent_request = intent_request
        self.input_transcript = self.intent_request['inputTranscript'] # user input
        self.session_attributes = utils.get_session_attributes(self.intent_request)
        self.fulfillment_state = 'Fulfilled'
        self.message = {'contentType': 'PlainText', 'content': ''}


    def dispatch_intent(self):
        # Set context with convo history for custom memory in langchain
        conv_context: str = self.session_attributes.get('ConversationContext', json.dumps(initial_history))

        logger.debug('Conv Conext:')
        logger.debug(conv_context)
        logger.debug(type(conv_context))

        # LLM
        openai_langchain_bot = OpenAILangchainBot(lex_conv_history=conv_context)
        llm_response = openai_langchain_bot.call_llm(user_input=self.input_transcript)
        print('llm_response :: ' + llm_response)
        
        self.message = {
            'contentType': 'PlainText',
            'content': llm_response
        }

        # save chat history as Lex session attributes
        session_conv_context = json.loads(conv_context)
        session_conv_context[CHAT_HISTORY] = \
            session_conv_context[CHAT_HISTORY] + self.input_transcript + \
            f'\nAI: {llm_response}' +'\nHuman: '

        self.session_attributes['ConversationContext'] = json.dumps(session_conv_context)

        self.response = utils.close(
            self.intent_request,
            self.session_attributes,
            self.fulfillment_state,
            self.message
        )

        return self.response
