from settings.settings import Settings
import openai
from loguru import logger


settings = Settings()


class AlfredBrain:

    @staticmethod
    async def think_about_it(prompt, token):
        logger.info('AlfredBrain: making a request to OpenAI API')
        logger.trace(f'AlfredBrain: making request to OpenAI with following prompt: {prompt}')
        openai.api_key = token
        try:
            response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    temperature=0.2,
                    max_tokens=100,
                    top_p=1,
                    n=1,
                    frequency_penalty=0.2,
                    presence_penalty=0,
                    stop=[" \n"]
            )
            resp = response["choices"][0]["text"]
            logger.trace(f'AlfredBrain: Got a response from OpenAI: {resp}')
        except Exception as ex:
            logger.debug(f'AlfredBrain: Did not manage to get or parse response from OpenAI,\
            exception raised: {ex}')
            resp = 'Sorry, I am a bit dizzy today. Could you repeat your request please?'

        return resp


alfred_brain = AlfredBrain()
