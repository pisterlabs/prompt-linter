""" AI Adapters by the langwizard to generate a response from different ai apis """
from app.openai_utils import openai_completion, openai_completion_stream
from app.ai21_utils import ai21_chat_completion


class AIAdapter:
    """ Base AI adapter for the langwizard """

    def __init__(self):
        raise NotImplementedError

    def chat(self, message):
        """ Returns a response from the ai given a message """
        raise NotImplementedError

    def chat_stream(self, message):
        """ Returns a reponse generator given a message """
        raise NotImplementedError


class OpenAIAdapter(AIAdapter):
    """ Adapter built for openai using the openai utils file """

    def __init__(self, openai_config):
        self.openai_config=openai_config

    def chat(self, message):
        return openai_completion(message,
                                 self.openai_config.MODEL_ENGINE,
                                 self.openai_config.MAX_RETRY)

    def chat_stream(self, message):
        return openai_completion_stream(message,
                                        self.openai_config.MODEL_ENGINE,
                                        self.openai_config.MAX_RETRY)


class Ai21Adapter(AIAdapter):
    """ Ai21 model adapter.  Specific to ai21 completion."""

    def __init__(self, ai21_config):
        self.ai21_config=ai21_config

    @staticmethod
    def _openai_like_to_prompt(message):
        return "\n".join([chat['content'] for chat in message])

    def chat(self, message):
        message = self._openai_like_to_prompt(message)
        return ai21_chat_completion(message, self.ai21_config.MODEL_ENGINE)

    def chat_stream(self, message):
        message = "\n".join([message_content['content'] for message_content in message])
        response = ai21_chat_completion(message, self.ai21_config.MODEL_ENGINE)
        # AI21 doesn't have a stream method on either their python sdk or api
        # The tokens are always presented with the full text in the response
        def reponse_generator(response):
            yield response['completions'][0]['data']['text'].strip()
        return reponse_generator(response)
