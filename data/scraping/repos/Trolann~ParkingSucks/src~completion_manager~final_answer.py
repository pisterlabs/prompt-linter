import newrelic.agent
from os import getenv
from langchain.chat_models import ChatOpenAI
from utils import logger, get_prompt, archive_completion

#TODO: Put back in answer_chains; this is broken out just for debugging the newrelic buffoonery

chat = ChatOpenAI(
    openai_api_key=getenv('OPENAI_API_KEY'),
    temperature=0.7,
    timeout=35
    )
application = newrelic.agent.register_application(timeout=0.1)
@newrelic.agent.background_task()
async def get_final_answer(question, schedule, closest_garages, gpt4=False) -> str:
    """
    This function gets all the available information and forms a response which will
    go directly to the user.
    :param question:
    :param schedule:
    :param closest_garages:
    :param gpt4:
    :return:
    """
    with newrelic.agent.BackgroundTask(application=application, name='get_final_answer'):
        logger.info(f'Getting final answer for question: {question}')
        question = await get_prompt(question, 'final', schedule=schedule, closest_garages=closest_garages)
        logger.debug(f'Prompt for final answer: {question}')
        chat.model_name = "gpt-3.5-turbo" if not gpt4 else "gpt-4"
        response = chat(question.to_messages())
        response = response.content

        await archive_completion(question.to_messages(), response)
        return response
