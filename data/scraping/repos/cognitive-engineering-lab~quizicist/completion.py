from math import ceil
from multiprocessing.dummy import Pool
import openai
import os
from dotenv import load_dotenv
from .consts import GPT_MODEL, MAX_CONTEXT_SIZE, ESTIMATED_QUESTION_SIZE, NUM_QUESTIONS
from .errors import QuizicistError
from .prompt import Prompt, PromptType
from .postprocess import postprocess_with_gpt, postprocess_manual

# set up openai
load_dotenv()
openai.api_key = os.getenv("OPENAI_SECRET_KEY")

# total number of tokens within a chapter
def chapter_tokens(components):
    return sum(map(lambda c: c["tokens"], components))


def shard_chapter(components):
    total_tokens = chapter_tokens(components)
    max_tokens = total_tokens / ceil(total_tokens / MAX_CONTEXT_SIZE)

    shards = []
    num_tokens = 0
    curr_prompt = ""

    for component in components:
        component_tokens = component["tokens"]
        component_text = component["text"]

        if num_tokens < max_tokens and num_tokens + component_tokens < MAX_CONTEXT_SIZE:
            curr_prompt += component_text
            num_tokens += component_tokens
        else:
            shards.append(curr_prompt)

            curr_prompt = component_text
            num_tokens = component_tokens

    shards.append(curr_prompt)
    return shards


def run_gpt3(shard, num_questions, custom_prompt, prompt_type):
    prompt = Prompt(custom_prompt=custom_prompt, prompt_type=prompt_type, num_questions=num_questions)\
        .add_system_prompt()\
        .add_message(role="user", content=shard)

    # process question until 5 well-formatted questions have been generated
    # TODO: add tally for failed generations and quit after n
    while True:
        print(f"Running completion on shard...")
        completion = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=prompt.messages, 
            max_tokens=NUM_QUESTIONS * ESTIMATED_QUESTION_SIZE,
            temperature=0.8,
        )
        
        print("Post processing shard...")
        processed = postprocess_with_gpt(
            completion["choices"][0]["message"]["content"],
            prompt.prompt_type,
            num_questions
        )

        if processed:
            return processed


# divide quiz questions evenly by shard
# don't allow more than five questions per shard
def divide_questions(shards, num_questions, custom_prompt, prompt_type):
    # find questions per shard and remainder after division
    remainder = num_questions % len(shards)
    questions_per_shard = num_questions // len(shards)

    jobs = []

    # handle case where more than five questions per shard
    if questions_per_shard > NUM_QUESTIONS or (questions_per_shard == NUM_QUESTIONS and remainder > 0):
        remaining_questions = num_questions - NUM_QUESTIONS * len(shards)

        jobs.extend(divide_questions(shards, remaining_questions, custom_prompt, prompt_type))
        jobs.extend(divide_questions(shards, num_questions - remaining_questions, custom_prompt, prompt_type))
    # handle normal case (max questions per shard <= 5)
    else:
        for index, shard in enumerate(shards):
            if index < remainder:
                jobs.append((shard, questions_per_shard + 1, custom_prompt, prompt_type))
            elif questions_per_shard > 0:
                jobs.append((shard, questions_per_shard, custom_prompt, prompt_type))

    return jobs

def complete(file_content, parser, num_questions, custom_prompt=None, prompt_type=PromptType.MCQ):
    components = parser(file_content)
    shards = shard_chapter(components)
    jobs = divide_questions(shards, num_questions, custom_prompt, prompt_type)

    # limit content size to three shards
    if len(shards) > 3:
        raise QuizicistError("Your uploaded content is too long. Please shorten the prompt and try again.")

    # parallelize GPT-3 calls
    with Pool(len(jobs)) as pool:
        return pool.starmap(run_gpt3, jobs)


def add_answer_choices(file_content, parser, question):
    components = parser(file_content)
    shard = shard_chapter(components)[question.shard]

    incomplete_question = f"""
Question: {question.question}
Correct answer: 
"""

    prompt = Prompt(prompt_type=PromptType.ADD_ANSWERS, num_questions=1)\
        .add_system_prompt()\
        .add_message(role="user", content=shard)\
        .add_message(role="user", content=incomplete_question)

    # TODO: clean up this loop or consolidate parsing into a single function
    while True:
        print("Running completion for custom question...")
        completion = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=prompt.messages,
            max_tokens=ESTIMATED_QUESTION_SIZE,
            temperature=0.8,
        )
                
        content = incomplete_question + completion["choices"][0]["message"]["content"]

        processed = postprocess_manual(content, question.shard)
        if processed:
            return processed

        print(f"Failed to parse the following:\n{completion}")
