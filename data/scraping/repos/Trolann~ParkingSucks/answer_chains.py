from final_answer import get_final_answer
from utils import get_prompt, archive_completion, logger, find_nearest_parking
from parking_chains import parking_chain
from os import getenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import OpenAIModerationChain
import newrelic.agent
import asyncio
import re

# Set up the chat model
chat = ChatOpenAI(
    openai_api_key=getenv('OPENAI_API_KEY'),
    temperature=0.7,
    timeout=35
    )

application = newrelic.agent.register_application(timeout=10.0)
@newrelic.agent.background_task()
async def answer_chain(username, question, message_id, user_id, memory, gpt4=False) -> str:
    """
    This function is the main function that is called by the discord bot to get a response to a question.
    It calls the moderation chain, then the gpt chain, then the final answer chain.
    :param username:
    :param question:
    :param message_id:
    :param user_id:
    :param memory:
    :param gpt4:
    :return:
    """
    # Complete GPT driven moderation
    schedule = memory.get_schedule(username)
    logger.debug(f'Got schedule of length {len(schedule)} for {username}')
    question = re.sub(r'<[^>]*>\s*', '', question)
    # This does moderation, gets parking data and gets location data at the same time. This is low cost
    is_ok_future, parking_info_future, map_info_future = await asyncio.gather(
        complete_gpt_moderation(question, username),
        parking_chain(question, schedule=schedule, gpt4=gpt4),
        map_chain(question, schedule=schedule, gpt4=gpt4),
    )

    # Assign a callback to process is_ok and decide whether to proceed with the final answer
    should_proceed, return_message = await process_is_ok(is_ok_future, username, question, memory)
    if not should_proceed:
        logger.critical(f'Got a bad question from {username}: {question}')
        return return_message

    parking_info = parking_info_future
    map_info = map_info_future
    logger.debug(f'Got parking info for {username}: {parking_info}')


    # Bad day
    if '%%%%%%%%' in map_info:
        logger.error(f'Unable to get map info for {username}')
        map_info = 'We were unable to extract any on-campus locations the user is talking about. Do not attempt to make up distances or use your knowledge of the location to determine closeness.'
    else: # Good day
        logger.debug(f'Got map info for {username}: {map_info}')
        map_info = f'Here are the disatnces to each garage from the location(s) the user is talking about: {map_info}'

    # Take all the data and get the final answer
    final_answer = await get_final_answer(question=question, schedule=parking_info, closest_garages=map_info, gpt4=gpt4)
    logger.debug(f'Ending answer chain for {username} with final answer: {final_answer}')
    return final_answer


@newrelic.agent.background_task()
async def complete_gpt_moderation(question, username) -> int:
    """
    Complete GPT driven moderation. Checks for unsafe content and ensures we're only talking
    about parking information or universities.
    :param question:
    :param username:
    :return:
    """
    question = await get_prompt(question, 'ok')
    chat.model_name = 'gpt-3.5-turbo'
    old_temp = chat.temperature
    chat.temperature = 0.2
    response = chat(question.to_messages()).content
    logger.debug(f'Got moderation response: {response}')
    chat.temperature = old_temp
    await archive_completion(question.to_messages(), response)
    logger.info(f'Got response: {response}')

    # Safe
    if '!!!!!!!!' in response:
        logger.debug(f'Found a safe question from {username}')
        return 1

    # Not sure
    if '########' in response:
        logger.warning(f'Found an unsure question from {username}')
        return 2

    # Unsafe
    if '@@@@@@@@' in response:
        logger.critical(f"UNSAFE QUESTION DETECTED FROM {username}:\n {question}")
    # Unable to parse
    else:
        logger.error(f'Unable to determine safety of query: {question}')
    return 0

@newrelic.agent.background_task()
async def map_chain(question, schedule, gpt4=False):
    """
    Determine location of places on campus in relation to garages.
    :param question:
    :param schedule:
    :param gpt4:
    :return:
    """
    logger.debug(f'Starting map chain for: {question}')

    # Get the question to feed to the model
    question = await get_prompt(question, 'map')

    # Temperature 0.2 leads to less hallucinations
    old_model_temp = chat.temperature
    chat.model_name = "gpt-3.5-turbo"# if not gpt4 else "gpt-4"
    chat.temperature = 0.2
    map_response = chat(question.to_messages())
    logger.debug(f'Got map response: {map_response}')
    # Cleanup
    chat.temperature = old_model_temp

    # This is likely for 'how's parking right now' or 'where can disabled people park'
    if '@@@@@@@@' in map_response.content:
        logger.debug(f'No map response needed for: {question}')
        return '%%%%%%%%'
    # Extract just the text after "!!!!!!!!"
    map_response = map_response.content.split('!!!!!!!!')[-1]
    logger.debug(f'Got map response: {map_response}')
    # Check if there is a comma in the string (multiple locations)
    if "," in map_response:
        # Split the string using the comma and strip any extra whitespace
        map_response_list = [item.strip() for item in map_response.split(',')]
    else:
        # If there is no comma, create a list with the single item
        map_response_list = [map_response]

    tables = ''
    for loc in map_response_list:
        tables += await find_nearest_parking(loc) + '\n'
    logger.debug(f'Completed map chain for: {question}')
    return tables

@newrelic.agent.background_task()
async def passed_moderation(query: str) -> bool:
    """
    Complete OpenAI required moderation. Checks for hateful content.
    :param query:
    :return:
    """
    try:
        moderation_chain = OpenAIModerationChain(error=True)
        moderation_chain.run(query)
        logger.info('OpenAI Moderation chain passed')
        return True
    except ValueError as e:
        logger.error(f"Flagged content detected: {e}")
        logger.error(f"Query: {query}")
        return False

async def process_is_ok(is_ok, username, question, memory):
    # Not allowed
    if is_ok == 0 or not await passed_moderation(question):
        logger.critical(f'Message not allowed: {question} (username: {username})')
        current_sched = memory.get_schedule(username)
        # Log bad actor
        if 'Bad Actor Count: ' not in current_sched:
            new_sched = f'{current_sched}\nBad Actor Count: 1'
            count = 1
        else:
            # Increment the number of bad actor counts
            count = int(current_sched.split('Bad Actor Count: ')[-1]) + 1
            # Remove old Bad Actor Count from current_shed to place a new one in
            current_sched = current_sched.split('Bad Actor Count: ')[0]
            new_sched = f'{current_sched}\nBad Actor Count: {count}'
        memory.write_schedule(username, new_sched)
        logger.debug(f'Wrote new schedule for {username}: {new_sched}')

        # determine first/second/third/found string based on count
        return False, 'You\'re not allowed to ask that.' if count == 1 else f"You're not allowed to ask that. I've had to tell you {count} times."

    # Couldn't determine if we can handle this, let's try anyway.
    if is_ok == 2:
        logger.warning(f'Trying a command, we shall see. Question: {question}')
    return True, ""


# Prompt 1:
# Add a flask api to this application. There should be 1 end-point: /completion
# which confirms the API_KEY environment variable matches the api_key argument.
# The 'query' argument should be passed to the 'complete_gpt_moderation' function and if it's ok,
# then to the 'get_sql_query' function. The /completion endpoint should return
# the result of get_sql_query.

# Prompt 2:
# Write python function(s) to call this api with the required parameters and
# decode the response. Update the parking-api.py file as needed.

# Prompt 3:
# <Gave Code>
# <Gave errors>
# <Gave API endpoint code>
# How can I properly decode final_answer in the discord bot to display it in the chat room?

# Prompt 4:
# This function:
# <gave get_sql_query>
# Is taking this response value:
# <Gave example response>
# and not properly returning the extracted SQL query. Fix it, please.

# Prompt 5:
# Take this query and use the CONCAT or similar function to add a single row to the top that says:
# Data generated for Tuesday and Thursday's from 12:30pm until 1:30pm.

# Prompt 6:
# How can I assign parking_info and map_info with simultaneous calls/threads? I'd like them to kick off at the same time but then wait for the values to be returned

# Prompt 6:
# Update this script to:
# Kick-off the assignment of is_ok, parking_info and map_info at the same time.
# Assign a callback to process is_ok and if it passes (returns non-0) does the send_reply and allows the final_answer call to kickoff when parking_info and map_info are done, and if it doesn't pass it blocks final_answer from being called and processes the bad actor.
#
# Give me the whole script in a codeblock. Explain what you did after the codeblock

