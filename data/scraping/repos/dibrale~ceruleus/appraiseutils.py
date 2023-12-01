import re
import asyncio
from asyncio import AbstractEventLoop

from langchain import LLMChain, LlamaCpp, PromptTemplate
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from modules.api import send_update
from modules.config import path, llm_params
from modules.rwutils import *
from modules.stringutils import list_or_str_to_str, bool_from_str, check_nil
from modules.tokenutils import llama_chunk, llama_token_length
from modules.saveutils import write_crumb


# Explicitly adapt the LangChain LLamaCpp wrapper to our parameters and initialize the LLM asynchronously
async def llama(loop: AbstractEventLoop):
    print_v("Initializing Llama model", params['verbose'])
    llm = await loop.run_in_executor(None, lambda: LlamaCpp(
            model_path=params['model_path'],
            callback_manager=BaseCallbackManager([StreamingStdOutCallbackHandler()]),
            verbose=params['verbose'],
            echo=True,
            **llm_params
        )
    )
    return llm


# General function for processing text through an LLM using a template
async def process(text: str | list[str], language_model: any, template_type: str = 'summarize') -> str:
    # Short circuit to empty output if input is empty:
    if check_nil(text):
        print_v(f'Empty input for {template_type} process - returning blank string')
        return ''

    # Load prompt
    prompt_template = await read_text_file(path[template_type + '_template'])

    # Prepare the input string, escape with a trite statement if none is provided
    input_text = list_or_str_to_str(text, if_blank='Nothing.')

    # Format the prompt using the input string
    # prompt = prompt_template.format(input_text=input_text)
    prompt = PromptTemplate.from_template(prompt_template)

    # Declare the chain, then run it and return output
    chain = LLMChain(
        llm=language_model,
        prompt=prompt,
        verbose=params['verbose'],
    )
    output = await asyncio.to_thread(chain, input_text,
                                     return_only_outputs=True,
                                     callbacks=BaseCallbackManager([StreamingStdOutCallbackHandler()]))
    print_v(f"Completed {template_type} process")
    return output['text']


# Process the input, splitting into chunks if necessary
async def chunk_process(
        text: str | list[str],
        language_model: any,
        template_type: str = 'summarize',
        max_tokens=1300,
        chunk_tokens=1024,
        output_list=False,
        summarize_chunks=True
) -> str | list[str]:

    input_text = list_or_str_to_str(text, if_blank='Nothing.')
    if llama_token_length(input_text) < max_tokens:
        out = await process(input_text, language_model, template_type)
        if output_list:
            return [out]
        return out

    processed_chunks = []
    text_chunks = llama_chunk(input_text, chunk_tokens)
    for chunk in text_chunks:
        processed_chunk = await process(chunk, language_model, template_type)
        processed_chunks.append(processed_chunk)

    if output_list:
        return processed_chunks
    if summarize_chunks:
        try:
            summary = await process(processed_chunks, language_model)
            return summary
        except ValueError as e:
            print_v(f"{e} - returning string of chunks instead")
    return list_or_str_to_str(processed_chunks)


# Function to determine whether a question was answered
async def is_answered(
        question: str,
        answer: str | list[str],
        language_model: any
) -> bool | None:

    # Listify the answer text
    if type(answer) is str:
        answer_list = [answer]
    else:
        answer_list = answer

    print_v("Checking if question has been answered")

    for candidate in answer_list:
        # Prepare the question and answer for input
        input_text = \
            f"Question: {question.strip()}\n\nAnswer: {list_or_str_to_str(candidate, if_blank='Nothing.').strip()}"

        # Serial execution only - for now
        await send_update('aye_start')
        aye = await process(input_text, language_model, 'aye')
        await send_update('nay_start')
        nay = await process(input_text, language_model, 'nay')

        # Initialize and run the judge
        judge_input = f"{input_text}\n\nArgument in favor: {aye}\n\n Argument against: {nay}"
        await send_update('judge_start')
        verdict_txt = await process(judge_input, language_model, 'judge')
        verdict = bool_from_str(assure_string(verdict_txt))
        await send_update('verdict_done')

        # Kill the function and return true if an answer is found
        if verdict:
            print_v(f"Question has been answered", not params['verbose'])
            print_v(f"Question has not been answered: {question}", params['verbose'])
            return True

    # Lament the lack of an answer for the log, then return False
    print_v(f"Question has not been answered", not params['verbose'])
    print_v(f"Question has not been answered: {question}", params['verbose'])
    return False


# Get thoughts related to the goal. Capable of batch execution using async code
async def goal_thoughts(
        goal: str,
        thought_list: list[str],
        language_model: any,
        loop: AbstractEventLoop,
        include_uncertain=False,
        n_batch=1
) -> list[str] | None:

    print_v("Gathering goal-related thoughts", params['verbose'])

    # Just give up there are no input thoughts
    if not thought_list:
        print_v("No thoughts. Returning empty list.")
        return []

    # Prepare the goal string
    goal_str = f"Goal: {goal.strip()}\n\n"

    # Prepare reply tasks
    reply_tasks = []
    print_v("Creating tasks for each thought", params['verbose'])
    for thought in thought_list:
        input_text = f"{goal_str}Thought: {thought.strip()}"
        reply_tasks.append(loop.create_task(process(input_text, language_model, 'goal_thought_eval')))

    # Asynchronously gather verdict batches
    str_verdicts = []
    for i in range(0, len(reply_tasks), n_batch):
        last_batch_element = min(i+n_batch, len(reply_tasks))-1
        if last_batch_element-i > 1:
            print_v(f"Processing next {last_batch_element-1} thoughts")
        elif last_batch_element-i == 1:
            print_v(f"Processing next thought")
        [new_verdicts] = await asyncio.gather(*reply_tasks[i:last_batch_element])
        str_verdicts.append(new_verdicts)

    # Sanity check before continuing
    try:
        assert len(thought_list) == len(str_verdicts)
    except AssertionError:
        print_v("AssertionError: Length mismatch between list of thoughts and verdicts. Returning 'None' and "
                "continuing.")
        return None

    # Aggregate and return pertinent thoughts
    pertinent_thoughts = []
    for i in range(len(thought_list)):
        if bool_from_str(str_verdicts[i]) or (include_uncertain and bool_from_str(str_verdicts[i]) is None):
            pertinent_thoughts.append(thought_list[i])
    print_v(f"{len(pertinent_thoughts)} thoughts are pertinent to the goal")
    return pertinent_thoughts


# Determine whether a goal has been met
async def goal_met(
        goal: str,
        thoughts: str | list[str],
        answers: str | list[str],
        language_model: any
) -> bool | None:
    await send_update('start')
    # Prepare goal, thoughts and answers for input
    goal_text = f"Goal: {goal.strip()}"
    thought_text = f"Thoughts: {list_or_str_to_str(thoughts.strip(), if_blank='None.').strip()}"
    answer_text = f"Answers: {list_or_str_to_str(answers.strip(), if_blank='None.').strip()}"
    input_text = goal_text + '\n\n' + thought_text + '\n\n' + answer_text

    # Ask the LLM if the goal has been achieved
    result = await chunk_process(input_text, language_model, 'goal_eval', output_list=True)

    # Parse the result and return a boolean, checking for inconsistency
    met_out = False
    for answer in result:
        # Parse the result and return a boolean, checking for inconsistency
        substring = re.search(
            r".*?Verdict\s*\d*\s*:\s*\d*\s*(?P<VERDICT>(.*?))$",
            answer
        )['VERDICT'].strip(' .?!,:<>()[]').lower()
        met = bool_from_str(assure_string(substring))
        if met:
            met_out = True
            break

    # Log output
    if met_out:
        modifier = ''
    else:
        modifier = 'not yet '

    print_v(f"Goal has {modifier}been met", not params['verbose'])
    print_v(f"Goal has {modifier}been met: {goal}", params['verbose'])
    await send_update('stop')

    return met_out


# Separate and return met and unmet goals
async def check_goals(
        goals: list[str],
        thoughts: str | list[str],
        answers: str | list[str],
        language_model: any,
        uncertain_is_met=False,
        write_crumbs=True
) -> {str: list[str]}:
    # Initialize the two buckets
    goals_met = []
    goals_unmet = []

    # Iterate through each goal and decide. No batch processing by design - it is available in goal_met instead
    print_v(f"Checking {len(goals)} goals")
    for goal in goals:
        met = await goal_met(goal, thoughts, answers, language_model)
        if met is None and uncertain_is_met:
            met = True
        if met:
            goals_met.append(goal)
            if write_crumbs:
                await send_update('goal_met')
                await write_crumb(
                    f"{goal}. There were some thoughts I had about this. {list_or_str_to_str(thoughts)}",
                    prefix=f"I was able to do something that I wanted to - ")
        else:
            goals_unmet.append(goal)

    # Log output and return
    print_v(f"Met {len(goals_met)}/{len(goals)} goals")
    return {'met': goals_met, 'unmet': goals_unmet}


# Check if the question currently written for Squire has been answered. If yes, write a crumb and clear the answers.
async def check_answered(answers: str | list[str], llm: any) -> [str, bool]:
    # Synthesize reply from list if necessary rather than just concatenating it
    if type(answers) is list[str]:
        answer = await chunk_process(answers, llm, 'synthesize')
    else:
        answer = answers
    question = await read_text_file(path['squire_question'])
    answered = await is_answered(question, answer, llm)

    if answered:
        crumb_task = write_crumb(
            f"I also thought of an answer to the question I asked. {answer}",
            prefix=f"I asked myself a question. {question}")
        clear_answers_task = write_json_data({'answers': []}, path['answers'])
        await asyncio.gather(crumb_task, clear_answers_task)
    return [question, answered]


# Set a new goal given a character description, some met goals and some failed goals
async def new_goal(
        met_goal_list: list[str],
        failed_goal_list: list[str],
        character_text: str,
        language_model: any
) -> str:
    await send_update('start')
    print_v("Generating a new goal", params['verbose'])
    input_lines = [f"Character Description:\n{character_text}"]
    bullet = f"\n- "
    if not check_nil(met_goal_list):
        try:
            input_lines.append(f"\nGoals met: {bullet.join(met_goal_list)}")
        except TypeError:
            pass
    if not check_nil(failed_goal_list):
        try:
            input_lines.append(f"\nGoals failed: {bullet.join(failed_goal_list)}")
        except TypeError:
            pass

    output = await process('\n\n'.join(input_lines), language_model, 'set_goal')
    await send_update('stop')
    return output
