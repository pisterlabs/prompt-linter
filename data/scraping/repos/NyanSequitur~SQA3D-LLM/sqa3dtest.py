# author: me :)
# date: 2021-07-31
# description: test the sqa3d dataset using gpt-3.5-turbo

# I don't like how this code handles reasoning
# it should not be caption-order dependent how the LLM generates reasoning
import logging
# log to log2.log
logging.basicConfig(filename='log2.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

# Imports
import os
from transformers import (
    AutoConfig,
    AutoTokenizer,
    set_seed,
)
import json

import re
import time
from typing import List
from tqdm.auto import tqdm

import tiktoken
import openai
from dotenv import load_dotenv
# load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# set constants
NUMBER_OF_QUESTIONS_TO_SOLVE = -1
CONFIRM_EVERY_N_QUESTIONS =-1
PAUSE_ON_FAIL = False
PAUSE_ON_FIRST = False
CACHE_MAX_AGE = 86400
MODEL_NAME = "tiiuae/falcon-40b-instruct"
openAIEngine = "gpt-3.5-turbo-16k"


# set up openai engine
encoder = tiktoken.encoding_for_model(openAIEngine)


if input("Delete savestate? (y/n) ").lower() == "y":
    try:
        os.remove("savestate.json")
    except FileNotFoundError:
        pass


def isWord(word):
    return word in words


# Reasoning format:
# Context:
# <context>
#
# Situation:
# <situation>
#
# Carefully work through every step of solving this question and briefly explain reasoning. Do not include the caption in the reasoning.
# <question>
#
# Reasoning:
# 1.
def generateReasoning(context, situation, question):
    prompt = f"Captions:\n{context}\n\nWork through this question step-by-step, very quickly explaining reasoning:\n{situation} {question}\n\nReasoning:\n1. "
    # write prompt to file
    with open("prompt.txt", "a") as f:
        f.write(prompt + "\n\n\n\n")
    return prompt
def getLLMResponse(
    model,
    prompt,
    max_new_tokens=100,
    temperature=1.9,
    repetition_penalty=1.2,
):
    # encode prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    # get number of tokens in prompt
    inputTokens = len(input_ids[0])
    # generate response
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        bad_words_ids=badWords,
    )
    # decode response
    outputTokens = len(outputs[0])
    answer = tokenizer.decode(outputs[0])

    # count number of tokens generated by subtracting the length of the prompt from the length of the answer
    print(f"Generated {outputTokens - inputTokens} tokens")

    return answer, outputTokens - inputTokens


# ______
# Question:
# <situation> <question>
# - - - - -
# Reasoning:
# <reasoning>
# - - - - -
# Answer:
# <answer>
def printResults(reasoning, answer, question, situation):
    print("_" * os.get_terminal_size().columns)
    print("Question:\n" + situation + " " + question)
    print("- " * (os.get_terminal_size().columns // 2))
    print("Reasoning:\n" + reasoning)
    print("- " * (os.get_terminal_size().columns // 2))
    print("Answer:\n" + answer)


# taken from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def sendMessage(context, situation, question, allowUnsure=True):
    message = f"{context}\n\n{situation} {question} Let's work through this step-by-step, before coming to the most likely one-word answer."
    if allowUnsure:
        message += " Answer 'unsure' if you are unsure."
    messages = [
        {
            "role": "system",
            "content": "You will be presented with a list of captions describing a room, followed by a question which is set in that room."  
        },
        {
            "role": "user",
            "content": message,
        }
    ]
    modelOutput = openai.ChatCompletion.create(
        model=openAIEngine,
        messages=messages,
        max_tokens=350 + num_tokens_from_messages(messages, model='gpt-3.5-turbo-0613'),
        temperature=0,
    )

    fullAnswer = modelOutput["choices"][0]["message"]["content"]
    #print(fullAnswer)
    answer = fullAnswer.split("\n")[-1]
    answer = answer.split(":")[1] if ":" in answer else answer.split()[-1]
    answer = "".join([c for c in answer if c.isalpha()])

    return answer, modelOutput


def promptOpenAI(context, question, situation, actuallyPrompt=True):
    # just for testing the rest of the code without using tokens
    if not actuallyPrompt:
        return ""

    answer, modelOutput = sendMessage(context, situation, question)
    tk = modelOutput["usage"]["completion_tokens"]
    
    if answer.strip().lower() == "unsure":
        # flip order of context
        # This is such a hacky way to do this
        # Ideally, the response is caption-order agnostic
        context = "\n".join(context.split("\n")[::-1])
        answer, modelOutput = sendMessage(context, situation, question, False)
    tk += modelOutput["usage"]["completion_tokens"]
    return answer, tk, modelOutput













unifiedQA=0
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "allenai/unifiedqa-v2-t5-large-1363200" # you can specify the model size here
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)


















# save state to file
def saveState(
    question,
    reasoning,
    answer,
    correct_answer,
    is_correct,
    token_count,
    totalTokenCount,
    answered,
    correct,
):
    try:
        f = open("savestate.json")
        try:
            savestate = json.load(f)
        except json.decoder.JSONDecodeError:
            savestate = {
                "answered": 0,
                "correct": 0,
                "totalTokenCount": 0,
                "questions": [],
            }
    except FileNotFoundError:
        savestate = {
            "answered": 0,
            "correct": 0,
            "totalTokenCount": 0,
            "questions": [],
        }
        # create file if it doesn't exist
        open("savestate.json", "w").close()

    savestate["answered"] = answered
    savestate["correct"] = correct
    savestate["totalTokenCount"] = totalTokenCount
    savestate["questions"].append(
        {
            "scene_id": question["scene_id"],
            "situation": question["situation"],
            "question": question["question"],
            "question_id": question["question_id"],
            # "relevant_nouns": question["relevantNouns"],
            "context": question["context"],
            "reasoning": reasoning,
            "answer": answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "token_count": token_count,
        }
    )
    with open("savestate.json", "w") as f:
        json.dump(savestate, f, indent=4)


# compile regex pattern to remove all non-alphanumeric characters
pattern = re.compile(r"[\W_]+")


# load /usr/share/dict/words into a set
with open("/usr/share/dict/words") as f:
    words = set(f.read().splitlines())

# load nouns_with_context.json
with open("nouns_with_context.json") as f:
    nouns = json.load(f)

# load answers
with open("v1_balanced_sqa_annotations_test_scannetv2.json") as f:
    answers = json.load(f)["annotations"]
answers_dict = {answer["question_id"]: answer for answer in answers}

answers = [answers_dict[noun["question_id"]] for noun in nouns]


if os.path.exists("savestate.json") and os.path.getsize("savestate.json") > 0:
    with open("savestate.json") as f:
        savestate = json.load(f)
    answered = savestate["answered"]
    correct = savestate["correct"]
    totalTokenCount = savestate["totalTokenCount"]
else:
    answered = 0
    correct = 0
    totalTokenCount = 0


# skip questions/answers that have already been answered
nouns = nouns[answered:]
answers = answers[answered:]
GTcorrect = 0

# reset answers.txt
open("answers.txt", "w").close()

generationTimes = []

is_correct = True

config = AutoConfig.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)
print("loaded config")
# model = AutoModelForCausalLM.from_pretrained(
#    MODEL_NAME,
#    trust_remote_code=True,
#    load_in_8bit=True,
#    device_map="balanced",
# )

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, add_prefix_space=True, trust_remote_code=True
)

badWordlist = ["<strong>"]
badWords = tokenizer(badWordlist, add_special_tokens=False).input_ids




for i, question in enumerate(tqdm(nouns, position=0, leave=True)):
    if i >= NUMBER_OF_QUESTIONS_TO_SOLVE and not NUMBER_OF_QUESTIONS_TO_SOLVE == -1:
        break
    # confirm every n questions, if the answer is wrong, and if it's the first question
    if (
        (i % CONFIRM_EVERY_N_QUESTIONS == 0 and not CONFIRM_EVERY_N_QUESTIONS == -1)
        or (not is_correct and PAUSE_ON_FAIL)
        or (i == 1 and PAUSE_ON_FIRST)
    ):
        if input("Continue? (y/n) ") != "y":
            break
    is_correct = True
    questionTokenCount = 0
    tempContext = "\n".join(question["context"])
    tempSituation = question["situation"]
    tempQuestion = question["question"]

    # reasoningPrompt = generateReasoning(tempContext, tempSituation, tempQuestion)
    #
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     start = time.time()
    #     reasoning, temp = getLLMResponse(
    #         model,
    #         reasoningPrompt,
    #         max_new_tokens=650,
    #         repetition_penalty=1.2,
    #         temperature=1,
    #     )
    #     end = time.time()
    #
    # questionTokenCount += temp
    # totalTokenCount += temp
    # if reasoning.endswith("<|endoftext|>"):
    #     reasoning = reasoning[: -len("<|endoftext|>")]
    # else:
    #    reasoning = reasoning[: reasoning.rfind("\n")]
    # reasoning = reasoning[len(reasoningPrompt) - len("1. ") :]
    # send reasoning to openai
    start2 = time.time()
    answer, temp, fullAnswer = promptOpenAI(tempContext, tempQuestion, tempSituation, True)
    end2 = time.time()
    questionTokenCount += int(temp)
    totalTokenCount += int(temp)
    generationTimes.append(((end2 - start2)))
    # write answer to file as new line
    with open("answers.txt", "a") as f:
        f.write(answer + "\n")
    answered += 1
    # answer should be just lowercase letters with no punctuation or spaces or newlines or anything
    answer = pattern.sub("", answer)

    # strip <|endoftext|> from reasoning
    # reasoning = reasoning.replace("<|endoftext|>", "")
    reasoning = fullAnswer["choices"][0]["message"]["content"]
    printResults(reasoning, answer, tempQuestion, tempSituation)

    if answer.lower() == answers[i]["answers"][0]["answer"].lower():
        correct += 1
    # check if last two words of reasoning are the same as the correct answer
    elif answer in answers[i]["answers"][0]["answer"].lower() and " ".join(reasoning.split()[-2:]).lower() == answers[i]["answers"][0]["answer"]:
        correct += 1
    else:
        is_correct = False
        # if answer isn't even a word, save it to a file
        if not isWord(answer.lower()):
            with open("badAnswers.txt", "a") as f:
                f.write(answer + "\n")
        print("Correct answer: " + answers[i]["answers"][0]["answer"])
    print(f"Answered: {answered}, Correct: {correct}, Accuracy: {round(correct / answered * 100, 2)}%")
    print(f"Question token count: {questionTokenCount}")
    print(f"Total token count: {totalTokenCount}")
    print(f"Average token count per question: {totalTokenCount / answered}")

    print(f"Average tokens per second: {totalTokenCount / sum(generationTimes)}")

    saveState(
        question,
        reasoning,
        answer,
        answers[i]["answers"][0]["answer"],
        answer.lower() == answers[i]["answers"][0]["answer"].lower(),
        questionTokenCount,
        totalTokenCount,
        answered,
        correct,
    )
    
    
    
    # Ground truth reasoning
    tempContext = "\n".join(question["gtCaptions"])
    
    answer, temp, fullAnswer = promptOpenAI(tempContext, tempQuestion, tempSituation, True)
    end2 = time.time()
    questionTokenCount += int(temp)
    totalTokenCount += int(temp)
    generationTimes.append(((end2 - start2)))
    # write answer to file as new line
    with open("GTanswers.txt", "a") as f:
        f.write(answer + "\n")
        
    if answer.lower() == answers[i]["answers"][0]["answer"].lower():
        GTcorrect += 1
    # check if last two words of reasoning are the same as the correct answer
    elif answer in answers[i]["answers"][0]["answer"].lower() and " ".join(reasoning.split()[-2:]).lower() == answers[i]["answers"][0]["answer"]:
        GTcorrect += 1
    else:
        is_correct = False
        # if answer isn't even a word, save it to a file
        if not isWord(answer.lower()):
            with open("badAnswers.txt", "a") as f:
                f.write(answer + "\n")
        print("Correct answer: " + answers[i]["answers"][0]["answer"])
        
        
        
        
    print(f"Ground Truth Captions | Answered: {answered}, Correct: {GTcorrect}, Accuracy: {round(GTcorrect / answered * 100, 2)}%")
    
    # unifiedQA
    tempContext = "\n".join(question["context"])
    if run_model(f"{tempContext}\n\n{tempSituation} {tempQuestion} Let's work through this step-by-step, before coming to the most likely one-word answer.")[0].lower() == answers[i]["answers"][0]["answer"].lower():
        unifiedQA += 1
    print(f"UnifiedQA | Answered: {answered}, Correct: {unifiedQA}, Accuracy: {round(unifiedQA / answered * 100, 2)}%")
    tempContext = "\n".join(question["gtCaptions"])
    if run_model(f"{tempContext}\n\n{tempSituation} {tempQuestion} Let's work through this step-by-step, before coming to the most likely one-word answer.")[0].lower() == answers[i]["answers"][0]["answer"].lower():
        unifiedQA += 1