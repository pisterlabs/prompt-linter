import openai
import logging
import constants
from utils import get_secret

logging.basicConfig(level=logging.INFO)


def ask_openai(user_input=None, previous_conversation_response=None):
    """

    :param user_input:                          user input
    :param previous_conversation_response:      prev conversation response in a list
    :return:                                    open ai response
    """
    last_response = previous_conversation_response[-1]
    try:
        openai.api_key = get_secret(constants.OPENAI_API_KEY)

        openai_response = openai.ChatCompletion.create(
                model=constants.openai_model,
                messages=[
                    {"role": "system", "content": f"{constants.initial_conversation}"},
                    {"role": "assistant", "content": f"{last_response}"},
                    {"role": "user", "content": f"{user_input}"},
                ]
            )

        previous_conversation_response.append(openai_response['choices'][0]['message']['content'])
        logging.info("After appending " + str(previous_conversation_response[-1]))

    except KeyError as e:
        logging.info('OpenAI API key not set.')
        exit(1)

    return openai_response['choices'][0]['message']['content'], previous_conversation_response
