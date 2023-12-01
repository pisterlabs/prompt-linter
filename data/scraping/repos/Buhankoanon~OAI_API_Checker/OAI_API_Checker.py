# -*- coding: utf-8 -*-
import openai
import requests
from datetime import datetime, timedelta
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import colorama
import logging
from math import ceil

colorama.init()

logging.basicConfig(filename='OAI_API_Checker_logs.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def log_and_print(message, log_level=logging.INFO):
    print(message)
    logging.log(log_level, message)    

def list_models(api_key):
    openai.api_key = api_key
    models = openai.Model.list()
    return [model.id for model in models['data']]

def filter_models(models, desired_models):
    return [model for model in models if model in desired_models]

def try_complete(api_key):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=1,
        messages=[{'role':'user', 'content': ''}]
    )

RED = "\033[31m"
YELLOW = "\033[33m"
#GREEN = "\033[32m"
#BLINK = "\033[5m"
RESET = "\033[0m"

def check_key(api_key, retry_count=3):
    result = f"{api_key}\n"
    has_gpt_4_32k = False
    model_ids = []
    errors = []

    models = list_models(api_key)
    filtered_models = filter_models(models, desired_models)

    if filtered_models:
        for model_id in filtered_models:
            result += f"  - {model_id}\n"
            model_ids.append(model_id)
    else:
        result += "  No desired models available.\n"
    
    has_gpt_4 = "gpt-4" in model_ids
    has_gpt_4_32k = "gpt-4-32k" in model_ids
    has_only_turbo = "gpt-3.5-turbo" in model_ids and not has_gpt_4
    
    try:
        for attempts in range(retry_count):
            try:
                try_complete(api_key)
                break
            except Exception as e:
                error_message = str(e)
                if "The server is overloaded or not ready yet" in error_message:
                    logging.info(f'Error encountered when generating a completion on attempt {attempts+1}: {error_message}. Retrying...')
                    time.sleep(5)
                    continue
                else:
                    raise e

    except Exception as e:
        error_message = str(e)
        if "You exceeded your current quota" in error_message:
            result += f"{YELLOW}  This key has exceeded its current quota{RESET}\n"
        elif "Your account is not active" in error_message:
            result += f"{RED} Error: Your account is not active, please check your billing details on our website.{RESET}\n"
        else:
            result += f"{RED} Unexpected Error at check_key: {error_message}{RESET}\n"
            errors.append((api_key, error_message))

    return result, has_gpt_4, has_gpt_4_32k, has_only_turbo, errors

def checkkeys(api_keys):
    working_gpt_4_keys = set()
    no_quota_gpt_4_keys = set()
    working_gpt_4_32k_keys = set()
    no_quota_gpt_4_32k_keys = set()
    working_only_turbo_keys = set()
    no_quota_only_turbo_keys = set()
    result = ''
    total_errors = []
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(check_key, api_key) for api_key in api_keys]

        for idx, future in enumerate(futures, start=1):
            result += f"API Key {idx}:\n"
            key = api_keys[idx - 1]
            try:
                key_result, has_gpt_4, has_gpt_4_32k, has_only_turbo, errors = future.result()
                total_errors.extend(errors)
                
                result += key_result

                if has_only_turbo and "Error" not in key_result and "This key has exceeded its current quota" not in key_result and "This key is invalid or revoked" not in key_result:
                    working_only_turbo_keys.add(key)
                if has_gpt_4 and not has_gpt_4_32k and "Error" not in key_result and "This key has exceeded its current quota" not in key_result and "This key is invalid or revoked" not in key_result:
                    working_gpt_4_keys.add(key)
                if has_gpt_4_32k and "Error" not in key_result and "This key has exceeded its current quota" not in key_result and "This key is invalid or revoked" not in key_result:
                    working_gpt_4_32k_keys.add(key)

                if has_only_turbo and "This key has exceeded its current quota" in key_result:
                    no_quota_only_turbo_keys.add(key)
                if has_gpt_4 and "This key has exceeded its current quota" in key_result:
                    no_quota_gpt_4_keys.add(key)
                if has_gpt_4_32k and "This key has exceeded its current quota" in key_result:
                    no_quota_gpt_4_32k_keys.add(key)

            except Exception as e:
                error_message = str(e)
                if "Incorrect API key provided" in error_message:
                    result += f"{key}\n"
                    result += f"{RED}  This key is invalid or revoked{RESET}\n"
                elif "You must be a member of an organization to use the API" in error_message:
                    result += f"{key}\n"
                    result += f"{RED} Error: You must be a member of an organization to use the API. Please contact us through our help center at help.openai.com.{RESET}\n"
                elif "This key is associated with a deactivated account" in error_message:
                    result += f"{key}\n"
                    result += f"{RED} Error: This key is associated with a deactivated account. If you feel this is an error, contact us through our help center at help.openai.com.{RESET}\n"
                else:
                    result += f"{RED} Unexpected Error at checkkeys: {error_message}{RESET}\n"
                    total_errors.append((api_keys[idx - 1], error_message))
            result += '\n'

    with open('turbo.txt', 'w') as f:
        if len(working_only_turbo_keys) > 0:
            f.write('Working API keys with GPT-3.5-Turbo model:\n')
            f.write('\n'.join(working_only_turbo_keys) + '\n\n')
        if len(no_quota_only_turbo_keys) > 0:    
            f.write('Valid API keys with GPT-3.5-Turbo model and no quota left:\n')
            f.write('\n'.join(no_quota_only_turbo_keys) + '\n\n')

    with open('gpt4.txt', 'w') as f:
        if len(working_gpt_4_32k_keys) > 0:
            f.write('Working API keys with GPT-4-32k model:\n')
            f.write('\n'.join(working_gpt_4_32k_keys) + '\n\n')
        if len(no_quota_gpt_4_32k_keys) > 0:
            f.write('Valid API keys with GPT-4-32k model and no quota left:\n')
            f.write('\n'.join(no_quota_gpt_4_32k_keys) + '\n\n')
        if len(working_gpt_4_keys) > 0:
            f.write('Working API keys with GPT-4 model:\n')
            f.write('\n'.join(working_gpt_4_keys) + '\n\n')
        if len(no_quota_gpt_4_keys) > 0:
            f.write('Valid API keys with GPT-4 model and no quota left:\n')
            f.write('\n'.join(no_quota_gpt_4_keys) + '\n\n')

    with open('unexpected_errors.txt', 'w') as f:
        for i, (api_key, error) in enumerate(total_errors, start=1):
            f.write(f"Error #{i}:\n")
            f.write(f"API Key: {api_key}\n")
            f.write(f"Error Message: {error}\n\n")

    result += f"\nNumber of working API keys with only 'gpt-3.5-turbo' model: {len(working_only_turbo_keys)}\n"
    for key in working_only_turbo_keys:
        result += f"{key}\n"
    result += f"\nNumber of working API keys with 'gpt-4' model: {len(working_gpt_4_keys)}\n"
    for key in working_gpt_4_keys:
        result += f"{key}\n"
    result += f"\nNumber of working API keys with 'gpt-4-32k' model: {len(working_gpt_4_32k_keys)}\n"
    for key in working_gpt_4_32k_keys:
        result += f"{key}\n"
    result += f"\nNumber of valid API keys with only 'gpt-3.5-turbo' model and NO quota left: {len(no_quota_only_turbo_keys)}\n"
    for key in no_quota_only_turbo_keys:
        result += f"{key}\n"
    result += f"\nNumber of valid API keys with 'gpt-4' model and NO quota left: {len(no_quota_gpt_4_keys)}\n"
    for key in no_quota_gpt_4_keys:
        result += f"{key}\n"
    result += f"\nNumber of valid API keys with 'gpt-4-32k' model and NO quota left: {len(no_quota_gpt_4_32k_keys)}\n"
    for key in no_quota_gpt_4_32k_keys:
        result += f"{key}\n"    
    
    return result

def animate_processing_request():
    while not processing_done:
        sys.stdout.write("\rProcessing... |")
        time.sleep(0.1)
        sys.stdout.write("\rProcessing... /")
        time.sleep(0.1)
        sys.stdout.write("\rProcessing... -")
        time.sleep(0.1)
        sys.stdout.write("\rProcessing... \\")
        time.sleep(0.1)
    sys.stdout.write("\rDone!          \n")

if __name__ == '__main__':
    api_keys = []
    desired_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-32k"]
    log_and_print("Enter the API keys (one key per line). Press Enter twice when you're done:")
    while True:
        try:
            api_key = input()
        except:
            break

        if not api_key:
            break
        api_keys.append(api_key.strip())

    processing_done = False
    animation_thread = threading.Thread(target=animate_processing_request)
    animation_thread.start()

    result = checkkeys(api_keys)

    processing_done = True
    animation_thread.join()
    
    log_and_print("\n" + result)

    input("Press Enter to exit...")