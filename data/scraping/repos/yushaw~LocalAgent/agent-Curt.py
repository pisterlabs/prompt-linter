import openai
import time
import os

# Author: Curt Kennedy. 
# Date: Apr 22, 2023.
# License: MIT
# Goal: Design a simple AI Agent with no dependencies!
# This AI will NOT run forever.  It is also safe since it doesn't have API access beyond the OpenAI API.
#
# Usage: Just set your MainObjective, InitialTask, OPENAI_API_KEY at a minimum.
#
# Tips: Feel free to play with the temperature and run over and over for different answers.
#
# Inspired from BabyAGI: https://github.com/yoheinakajima/babyagi
# BabyAGI has many more features and bells and whistles.  But may be hard to understand for beginners.

# Goal configuration
MainObjective = "Become a machine learning expert." # overall objective
InitialTask = "Learn about tensors." # first task to research

# API Key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Note: As expected, GPT-4 gives much deeper answers.  But turbo is selected here as the default, so as there no cost surprises.
OPENAI_API_MODEL = "gpt-3.5-turbo" # use "gpt-4" or "gpt-3.5-turbo"

# Model configuration
OPENAI_TEMPERATURE = 0.7

# Max tokens that the model can output per completion
OPENAI_MAX_TOKENS = 1024

# init OpenAI Python SDK
openai.api_key = OPENAI_API_KEY


# print objective
print("*****OBJECTIVE*****")
print(f"{MainObjective}")


# dump task array to string
# 生成一个 单独的包含所有字符串的 string
def dumpTask(task):
    d = "" # init
    for tasklet in task:
        d += f"\n{tasklet.get('task_name','')}"
    d = d.strip()
    return d


# inference using OpenAI API, with error throws and backoffs
# 这里主要是创建 chat 并处理异常情况
def OpenAiInference(
    prompt: str,
    model: str = OPENAI_API_MODEL,
    temperature: float = OPENAI_TEMPERATURE,
    max_tokens: int = 1024,
):
    while True:
        try:
            # Use chat completion API
            response = "NOTHING"
            messages = [{"role": "system", "content": prompt}]
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                stop=None,
            )
            return response.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            print(
                "   *** The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.Timeout:
            print(
                "   *** OpenAI API timeout occured. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIError:
            print(
                "   *** OpenAI API error occured. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIConnectionError:
            print(
                "   *** OpenAI API connection error occured. Check your network settings, proxy configuration, SSL certificates, or firewall rules. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.InvalidRequestError:
            print(
                "   *** OpenAI API invalid request. Check the documentation for the specific API method you are calling and make sure you are sending valid and complete parameters. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.ServiceUnavailableError:
            print(
                "   *** OpenAI API service unavailable. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        finally:
            pass
            # print(f"Inference Response: {response}")

# expound on the main objective given a task
def ExpoundTask(MainObjective: str, CurrentTask: str):

    print(f"****Expounding based on task:**** {CurrentTask}")

    prompt=(f"You are an AI who performs one task based on the following objective: {MainObjective}\n"
            f"Your task: {CurrentTask}\nResponse:")


    # print("################")
    # print(prompt)
    response = OpenAiInference(prompt, OPENAI_API_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS)
    new_tasks = response.split("\n") if "\n" in response else [response]
    return [{"task_name": task_name} for task_name in new_tasks]



# generate a bunch of tasks based on the main objective and the current task
def GenerateTasks(MainObjective: str, TaskExpansion: str):
    prompt=(f"You are an AI who creates tasks based on the following MAIN OBJECTIVE: {MainObjective}\n"
            f"Create tasks pertaining directly to your previous research here:\n"
            f"{TaskExpansion}\nResponse:")
    response = OpenAiInference(prompt, OPENAI_API_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS)
    new_tasks = response.split("\n") if "\n" in response else [response]
    task_list = [{"task_name": task_name} for task_name in new_tasks]
    new_tasks_list = []
    for task_item in task_list:
        # print(task_item)
        task_description = task_item.get("task_name")
        if task_description:
            # print(task_description)
            task_parts = task_description.strip().split(".", 1)
            # print(task_parts)
            if len(task_parts) == 2:
                new_task = task_parts[1].strip()
                new_tasks_list.append(new_task)

    return new_tasks_list

# Simple version here, just generate tasks based on the inital task and objective, \
    # then expound with GPT against the main objective and the newly generated tasks.
# 生成目标的描述
q = ExpoundTask(MainObjective,InitialTask)
ExpoundedInitialTask = dumpTask(q)

print(ExpoundedInitialTask)

# 根据当前的任务，生成任务清单
q = GenerateTasks(MainObjective, ExpoundedInitialTask)
print(q)

TaskCounter = 0
for Task in q:
    TaskCounter += 1
    print(f"#### ({TaskCounter}) Generated Task ####")
    e = ExpoundTask(MainObjective,Task)
    print(dumpTask(e))