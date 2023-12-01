#!/usr/bin/env python3
import json
import csv
import sys
import argparse
from tqdm import tqdm
from time import sleep
from dotenv import dotenv_values
import openai
from nltk.tokenize.treebank import TreebankWordDetokenizer
from typing import List, Tuple
import re

defaults={
    'temperature': 0.1,
    'token_radius': 75,
}

configs=argparse.Namespace()

def main():
    get_args() 

    with configs.file as f:
        train_json = list(f)

    if configs.bootstrap:
        # load secrets
        secrets = dotenv_values(".env")
        openai.organization = secrets['organization']
        openai.api_key      = secrets['api_key']

    with configs.output as outfile:
        # write header
        writer = csv.writer(outfile)
        field = ['docid', 'text', 'triggers']
        writer.writerow(field)

        # load kairos roles
        global kairos_roles
        with configs.kairos as f:
            kairos_roles = json.load(f)

        # parse each line of jsonl file
        start_idx = configs.start
        end_idx = configs.end if configs.end != -1 else len(train_json) 

        for line in tqdm(train_json[start_idx:(end_idx+1)]):
            example = json.loads(line.encode('utf-8'))
            tokens = example['tokens']
            docid = example['doc_id']

            # convert events to metatriggers
            metatriggers = [event_to_metatrigger(event) for event in example['event_mentions']]

            # construct bootstrap pairs with RAW tokens
            if configs.bootstrap:
                add_bootstrap(tokens, metatriggers)

            # add markup about triggers and their indices to tokens
            add_indices(tokens, metatriggers)
            indexed_text = detokenize_and_collapsews(tokens)

            writer.writerow([docid, indexed_text, json.dumps(metatriggers)])

def event_to_metatrigger(event) -> dict:
    """Converts an event to a metatrigger (trigger with additional info to aid QA gen) """
    template, role_text_map = get_template_and_role_mapping(event)
    event_id = event['id']
    trigger = event['trigger']
    text = trigger['text']
    offset = trigger['start']
    # if debug:
    #     print(f"text:{text} offset:{offset} template:{template} role_text_map:{role_text_map}")
    return {'event_id': event_id, 'text':text, 'offset':offset, 'template':template, 'role_text_map':role_text_map}

def get_template_and_role_mapping(event) -> Tuple[str, List[dict]]:
    """Returns the template and role mapping for the given event"""
    role_mapping_dict = {arg['role']:arg['text'] for arg in event['arguments']}
    event_type = event['event_type']

    event_role = kairos_roles.get(event_type)
    if event_role is None:
        # retry with 'unspecified' variant of event type
        event_type = event_type.rsplit('.', 1)[0] + '.Unspecified' 
        event_role = kairos_roles.get(event_type)
        if event_role is None:
            return None, [] # just give up

    template = event_role['template']
    role_text_map = [{'role':role, 'text':role_mapping_dict.get(role)} for role in event_role['roles']]
    return template, role_text_map

def add_bootstrap(raw_tokens: List[str], triggers: List[dict]):
    """Adds bootstrap QA pairs to each trigger in the list of triggers"""
    for i, trigger in enumerate(triggers):
        offset = trigger['offset']

        # select region around trigger
        tkn_range_start = max(0, offset - configs.token_radius)
        tkn_range_end = min(len(raw_tokens), offset + configs.token_radius)
        
        # mark trigger and get text
        tokens = raw_tokens[tkn_range_start:tkn_range_end]
        tokens[offset-tkn_range_start] = f"<trigger> {trigger['text']} </trigger>"
        text = detokenize_and_collapsews(tokens)

        # declare messages
        msgs = [
            usr_msg("You annotate passages with question and answer pairs for the trigger."),
            *instruction_msgs,
            usr_msg(f"###\n Trigger: {trigger['text']}\n Passage: {text}"),
        ]

        # get response from GPT-3.5
        response = make_gpt_request(msgs)
        if configs.debug:
            def indent(s): 
                return '\n'.join('  ' + line for line in s.split('\n'))
            print("GOT RESPONSE FOR ###", text, "###\nTRIGGER: ", trigger)
            print(indent(response))

        # add boostrap to trigger
        triggers[i]['bootstrap'] = response

def make_gpt_request(msgs: List[dict]) -> str:
    """Makes a request to GPT-3.5 and returns the response"""
    # loop until request is fulfilled successfully
    while True:
        # prevent exceeding request limit per second
        sleep(0.1)
        try:
            # get response from GPT-3.5
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo-0301',
                messages=msgs,
                temperature=configs.temperature
            )
            return response['choices'][0]['message']['content']
        except openai.error.Timeout as e:
            print(f"OpenAI API request timed out: {e}")
        except openai.error.APIError as e:
            print(f"OpenAI API returned an API Error: {e}")
        except openai.error.APIConnectionError as e:
            print(f"OpenAI API request failed to connect: {e}")
        except openai.error.InvalidRequestError as e:
            print(f"OpenAI API request was invalid: {e}")
        except openai.error.AuthenticationError as e:
            print(f"OpenAI API request was not authorized: {e}")
        except openai.error.PermissionError as e:
            print(f"OpenAI API request was not permitted: {e}")
        except openai.error.RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}")
            sleep(1)
        except openai.error.ServiceUnavailableError as e:
            print(f"OpenAI API request had unavailable service")
            sleep(1)

def detokenize_and_collapsews(tokens: List[str]) -> str:
    """Detokenizes tokens and collapses whitespace in an intuitive fashion"""
    text = TreebankWordDetokenizer().detokenize(tokens)
    text = re.sub('\s*,\s*', ', ', text)
    text = re.sub('\s*\.\s*', '. ', text)
    text = re.sub('\s*\?\s*', '? ', text)
    text = re.sub('\s*\-\s*', '-', text)
    text = re.sub('\s*\’\s*', '’', text)
    text = re.sub('\s*\“\s*', ' “', text)
    text = re.sub('\s*\”\s*', '” ', text)
    text = re.sub('\s*\–\s*', '–', text)
    text = re.sub('\s*\'\s*', "'", text)
    text = re.sub('\s*\"\s*', '"', text)
    return text

def get_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        prog='parse_multitrigger.py', fromfile_prefix_chars='@',
        description='Parse data from jsonl file into csv format with one document per row.')
    parser.add_argument('file', type=argparse.FileType("r"), 
                        help='jsonl file to parse', metavar='src')
    parser.add_argument('-k', '--kairos', type=argparse.FileType("r"),
                        default='./datasets/event_role_formatted.json', help='kairos roles file')
    parser.add_argument('-o', '--output', type=csv_opener("w"), 
                        default='./datasets/mturk_data.csv', help='output file')
    parser.add_argument('-s', '--start', type=int, default=0, 
                        help='start index (inclusive) of range to parse in file; default: beginning of file')
    parser.add_argument('-e', '--end', type=int, default=-1, 
                        help='end index (inclusive) of range to parse in file; default: end of file')
    parser.add_argument('-t', '--temperature', type=float, 
                        default=defaults['temperature'], dest='temperature',
                        help='temperature parameter for GPT-3.5 for bootstrap; default: ' + str(defaults['temperature']));
    parser.add_argument('-r', '--radius', type=int, 
                        default=defaults['token_radius'], dest='token_radius',
                        help='radius of tokens around trigger to include in bootstrap; default: ' + str(defaults['token_radius']));
    parser.add_argument('-b', '--bootstrap', action='store_true', 
                        help='add bootstrap QA pairs to triggers')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='set debugging on')
    parser.parse_args(namespace=configs)

def csv_opener(mode: str):
    """Returns a function that opens a csv file in the specified mode """
    def opencsv(path: str) :
        """Opens a csv file in the specified mode"""
        try:
            return open(path, mode=mode, newline='', encoding='utf-8')
        except FileNotFoundError:
            raise argparse.ArgumentError(f"File {path} not found")
    return opencsv

def add_indices(raw_tokens: List[str], triggers: List[dict]):
    """Marks each trigger with an HTML tag"""
    for i, t in enumerate(triggers):
        o = t['offset']
        raw_tokens[o] = f"<trigger id='trigger-{i}'>{raw_tokens[o]}</trigger>"

def usr_msg(msg: str): return {"role": "user",       "content": msg}
def sys_msg(msg: str): return {"role": "system",     "content": msg}
def bot_msg(msg: str): return {"role": "assistant",  "content": msg}

example_msgs = [
    usr_msg("Here is an example.\n "),
    usr_msg(
    "Trigger: detonated\n Passage: "
    'A settlement has been reached in a $1-million lawsuit filed by a taxi driver accusing police of negligence after he got caught up in the August 2016 take-down of ISIS-sympathizer Aaron Driver.'
    'READ MORE: FBI agent whose tip thwarted 2016 ISIS attack in Ontario says he was glad to help.'
    ' Terry Duffield was injured when Driver <trigger> detonated </trigger> a homemade explosive in the back of his cab in August 2016.'
    ),
    bot_msg(
    """
Question: Who detonated the bomb?
Answer: Aaron Driver/ISIS-sympathizer

Question: What was detonated?
Answer: a homemade explosive

Question: When was the detonation?
Answer: Aug. 10, 2016

Question: Who did the detonation hurt?
Answer: Terry Duffield/taxi driver

Question: Where was the detonation?
Answer: taxi/Ontario
    """),
    usr_msg("Here is another example.\n "),
    usr_msg(
    "Trigger: inspections\n Passage: "
    "Amid speculation that President Bush is reconsidering what will constitute \"regime change in Iraq,\" one thing should be clear: Saddam Hussein's willingness to \"change\" his attitude towards permitting the resumption of intrusive on-site U.N. weapons <trigger> inspections </trigger> will not, in fact, eliminate the danger posed by him and his ruling clique. "
    "Indeed, what would be, at best, an ephemeral attitudinal adjustment on Saddam's part would probably not even diminish meaningfully the threat from Iraq's weapons of mass destruction programs. "
    "After all, were Saddam -- against all odds and past practice -- actually to cooperate with the U.N. inspectors and assist in the complete elimination of his chemical and biological arsenals, he could resume covertly stockpiling them again in as little as six-months time."
    ),
    bot_msg(
    """
Question: Who is inspecting something?
Answer: U.N.

Question: Who is being inspected?
Answer: Saddam Hussein

Question: What is being inspected?
Answer: weapons of mass destruction

Question: Where are the inspections?
Answer: Iraq
    """
    )
]

# create informational msgs for gpt
instruction_msgs = [
    usr_msg(
    "You are an assistant that reads through a passage and provides all possible question and answer pairs to the word with trigger tags. "
    "The word with the <trigger> tag around itis the event trigger, and the questions will help ascertain facts about the event."
    ),
    *example_msgs,
    usr_msg(
    "VERY IMPORTANT: Answers MUST be direct quotes from the passage. "
    "Answers should be accurate and the most informative possible. \n"
    "VERY IMPORTANT: Questions MUST be in the form: (Who/What/When/Where/Why/How) [verb] [subject] <trigger> [object] [preposition] [object] \n"
    ),
]


if __name__ == '__main__':
    main()
