import sys
from dotenv import load_dotenv
import json
import os
from transformers import GPT2TokenizerFast
from datetime import datetime, date, timedelta
from pathlib import Path
import re
from helpers import convert_messages_to_threads, generate_html_page, get_token_count, format_messages, get_today_str, get_one_week_before_ref_date, check_file_exists, check_matching_file_exists_ignore_creation_date, gen_and_get_discord_export, group_messages, flatten_thread, merge_consecutive_threads_by_same_author, sort_messages_by_timestamp, format_single_message, prompt_summarize_conversation_thread, extract_emoji_info, prompt_consolidate_bullet_summaries, prompt_get_consolidate_mappings, aggregate_reactions, extract_objects_by_ids, prompt_reformat_summary, create_chat_completion_with_retry
from constants import DISCORD_EXPORT_DIR_PATH, DISCORD_EXPORT_DIR_PATH_RAW, DISCORD_TOKEN_ID, CHANNEL_AND_THREAD_IDS, COMPLETIONS_MODEL, COMPLETIONS_API_PARAMS
import openai
load_dotenv()

# top level arguments 
arguments = {
    "file_type": 'json', # json or htmldark
    "channel_key": 'cabin_general_chat',
    "reference_date": '2023-07-12', #TODO if you enter today's date it will run for the last week 
    "force_file_regen": False
}

should_run_parts = {
    "generate_threads": False,
    "generate_summaries": False,
    "generate_summaries_insert_str": False,
    "generate_consolidate_mappings": False, # NOTE: i've turned this off for now because it's not working great
    "do_consolidation_and_reformat": False,
    "generate_html_bulletin": True,
}

# my prompts 
PROMPTS = {
    "summarize_conversation_thread": prompt_summarize_conversation_thread, # probably called first
    "consolidate_bullet_summaries": prompt_consolidate_bullet_summaries, # third
    "get_consolidate_mappings": prompt_get_consolidate_mappings, # second,
    "reformat_summary": prompt_reformat_summary # fourth
}

# create discord export 
file_name, file_type = gen_and_get_discord_export(
    DISCORD_EXPORT_DIR_PATH, 
    DISCORD_TOKEN_ID, 
    arguments['channel_key'], 
    arguments['file_type'],
    get_one_week_before_ref_date(reference_date=arguments['reference_date']), 
    arguments['reference_date'], 
    arguments['force_file_regen']
)

if should_run_parts['generate_threads']:
    # read file as json if json, otherwise exit if html
    if file_type == 'json':
        with open(DISCORD_EXPORT_DIR_PATH_RAW + '/' + file_name)  as f:
            discord_message_data = json.load(f)

            formatted_final = convert_messages_to_threads(discord_message_data)

            # Write the threads to a JSON file
            threads_file_path = DISCORD_EXPORT_DIR_PATH_RAW + '/' + 'last_run_threads.json' 
            with open(threads_file_path, 'w') as f:
                json.dump(formatted_final, f, indent=4)

if should_run_parts['generate_summaries']:
    all_summaries = {} 

    for index, thread in enumerate(formatted_final):

        all_summaries[index] = {
            "thread": thread,
            "summary": None,
            "reactions": extract_emoji_info(thread)
        }

        insert_discord_msgs_str = ""

        for message in thread:
            insert_discord_msgs_str += json.dumps(message) + "\n"

        prompt = PROMPTS['summarize_conversation_thread'](insert_discord_msgs_str)
        # print(f"\nCHANNEL KEY: {channel_key}\nPROMPT:\n{prompt}\n")
        print(prompt)

        # call api
        response = create_chat_completion_with_retry(prompt)

        all_summaries[index]['summary'] = response.choices[0].message.content

        print(response)

    # output all_summaries to file
    with open(DISCORD_EXPORT_DIR_PATH_RAW + '/' + 'last_run_summaries.json', 'w') as f:
        json.dump(all_summaries, f, indent=4)

# load last run summaries if they exist and create a str with them all 
if (should_run_parts['generate_summaries_insert_str']):
    # load last_run_summaries.json if it exists
    last_run_summaries_file_path = DISCORD_EXPORT_DIR_PATH_RAW + '/' + 'last_run_summaries.json'
    last_run_summaries = None
    with open(last_run_summaries_file_path) as f:
        last_run_summaries = json.load(f)
    # iterate through the dictionary last_run_summaries and append the summary of each one to a string containing all the summaries
    last_run_summaries_str = ''
    for key, value in last_run_summaries.items():
        # HACK: ignore things in this list, because i had too many items in the last_run_summaries.json, so these things are manually took out 
        # list for 6/9 run
        # ignore_list = ["1", "6", "12", "23", "29", "37", "40", "47", "60", "62", "67", "68", "69", "70", "71", "72"] 
        # list for 6/16 run
        ignore_list = []
        if key in ignore_list:
            continue
        # last_run_summaries_str += "ID: " + key + '\n'+ value["summary"] + '\n\n'
        # rewrite last_run_summaries_str with f formatting 
        last_run_summaries_str += f"ID: {key}\nReactions: {value['reactions']}\n{value['summary']}\n\n"

# generate consolidate mappings json
# NOTE: i'm going to forgo the consolidation mappings for now, they haven't been working great and also been having issues with it not giving me back just json
if (should_run_parts['generate_consolidate_mappings']):
    # create prompt
    prompt = PROMPTS["get_consolidate_mappings"](last_run_summaries_str)
    print(prompt)

    response = openai.ChatCompletion.create(messages=[{"role": "user", "content": prompt}], **COMPLETIONS_API_PARAMS)
    print(response)

    # get content of response
    # response_content = response['choices'][0]['content']
    response_content = response.choices[0].message.content

    # output to file json 
    output_file_path = DISCORD_EXPORT_DIR_PATH_RAW + '/' + 'last_run_consolidate_mappings.json'
    with open(output_file_path, 'w') as f:
        json.dump(json.loads(response_content), f, indent=4)

# load consolidate mappings and do consolidate and regernate the FINAL_SUMMARIES
if (should_run_parts['do_consolidation_and_reformat']):
    # load last_run_consolidate_mappings.json
    last_run_consolidate_mappings_file_path = DISCORD_EXPORT_DIR_PATH_RAW + '/' + 'last_run_consolidate_mappings.json'
    last_run_consolidate_mappings = None
    with open(last_run_consolidate_mappings_file_path) as f:
        last_run_consolidate_mappings = json.loads(f.read())

    # iterate through each lasst_run_consolidate_mappings and replace the summary with the consolidated summary
    for key, value in last_run_consolidate_mappings.items():
        consolidate_ids = value["consolidate_ids"]
        consolidate_reasoning = value["reasoning"]

        temp = extract_objects_by_ids(last_run_summaries, [str(id) for id in consolidate_ids])
        agg_re = aggregate_reactions(temp)

        # get the summaries from last_run_summaries
        # consolidate_summaries = []
        consolidate_summaries_str = ''
        for id in consolidate_ids:
            key = str(id)
            value = last_run_summaries[key]
            consolidate_summaries_str += f"ID: {key}\nReactions: {value['reactions']}\n{value['summary']}\n\n"
            # consolidate_summaries.append(last_run_summaries[str(id)]["summary"])

        # join the summaries with a newline
        # consolidate_summaries_str = '\n\n'.join(consolidate_summaries)

        # cann open ai to get the consolidated summary
        prompt = PROMPTS["consolidate_bullet_summaries"](consolidate_summaries_str)
        print(prompt)
        # get response 
        response = openai.ChatCompletion.create(messages=[{"role": "user", "content": prompt}], **COMPLETIONS_API_PARAMS)

        # remove the objects by key in last_run_summaries
        for id in consolidate_ids:
            del last_run_summaries[str(id)]

        # add the new consolidated summary to last_run_summaries using a combined key
        combined_key = '-'.join([str(id) for id in consolidate_ids])
        last_run_summaries[combined_key] = {
            "summary": response.choices[0].message.content,
            "reactions": agg_re,
            "consolidation_info": {
                "consolidate_ids": consolidate_ids,
                "reasoning": consolidate_reasoning 
            }
        }

        # Assuming last_run_summaries is a dictionary object
        print(json.dumps(last_run_summaries[combined_key], indent=4))

    # iterate through last_run_summaries and remove thread key and corresponding object if key exists 
    for key, value in last_run_summaries.items():
        if "thread" in value:
            del last_run_summaries[key]["thread"]

    # write last_run_summaries to new file called FINAL_last_run_summaries.json
    output_file_path = DISCORD_EXPORT_DIR_PATH_RAW + '/' + 'FINAL_last_run_summaries.json'
    with open(output_file_path, 'w') as f:
        json.dump(last_run_summaries, f, indent=4)

    for key, value in last_run_summaries.items():
        summary = value["summary"]
        prompt = PROMPTS["reformat_summary"](summary)
        print(prompt)
        response = openai.ChatCompletion.create(messages=[{"role": "user", "content": prompt}], **COMPLETIONS_API_PARAMS)
        print(response)
        last_run_summaries[key]["summary"] = response.choices[0].message.content

    # write last_run_summaries to new file called FINAL_FORMATTED_last_run_summaries.json
    output_file_path = DISCORD_EXPORT_DIR_PATH_RAW + '/' + 'FINAL_FORMATTED_last_run_summaries.json'
    with open(output_file_path, 'w') as f:
        json.dump(last_run_summaries, f, indent=4)

# generate html page
if (should_run_parts['generate_html_bulletin']):
    generate_html_page(arguments['channel_key'], arguments['reference_date'])

