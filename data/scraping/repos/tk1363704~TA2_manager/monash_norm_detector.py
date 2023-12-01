
# Very simple example of a TA1 process.
# Any pointing gesture is reported as offensive.
#
# This program subscribes to the GESTURE message queue, and looks for a
# pointing gesture.  If it sees one, it creates an "offensive" message
# and sends it on the RESULT message queue.


import logging
import zmq
import os
import time
import math
import openai
import random

from ccu import CCU

logging.basicConfig(datefmt='%Y-%m-%d %I:%M:%S',
                    format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG)
logging.info('Norm detection model is running...')

def detect_norm(in_message):
    if 'asr_text' in in_message:
        # # testing sample
        # if '我又没盯着你看' in in_message['asr_text']:
        #     return "apology", in_message['asr_text'], 'Yes', 0.5
        return in_context_gpt3(in_message['asr_text'])
    return "none", "none", "No", 0.0


def in_context_gpt3(input_text, prompt=''):
    '''input: a couple of sentences.
       output: None means no norm is detected. Or return 4 teams:
       1. Norm category: e.g. applogy
       2. utterances contain the norm
       3. Yes or No.  means wheather or not violate the norm
       4. The probability of violation. A smaller number means less probability of violation.

       Example of input: There is a seminar tomorrow morning. You should set an early alarm so that you can catch up it. Thanks for your reminder.
       Example of output: 'inform;', '"There is a seminar tomorrow morning. You should set an early alarm so that you can catch up it."', 'No', 0.0005
    '''
    keys = ["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",
            "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",
            "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",
            "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",
            "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"]
    # openai.api_key = os.getenv(random.choice(keys))
    cnt = 0
    response = ''
    while cnt < 100:
        cnt += 1
        try:
            response = openai.Completion.create(
                model="text-davinci-002",
                api_key=random.choice(keys),
                prompt="Given the dialogue:\n"
                       "你穿哪个衣服好难看，还是放弃吧。 你为啥这么说，我很伤心\n"
                       "Which social norm does the above dialogue contain?  Which does utterance show the social norm? Is there any norm violation?\n"
                       "none; apology; greeting; criticism; request; persuasion;"
                       "\n"
                       "criticism;\n\"你穿哪个衣服好难看，还是放弃吧\"\nYes\n"
                       "Given the dialogue:\n"
                       + input_text +
                       "Which social norm does the above dialogue contain?  Which does utterance show the social norm? Is there any norm violation?\n"
                       "\nnone; apology; greeting; criticism; request; persuasion;",
                temperature=0,
                max_tokens=256,
                top_p=1,
                logprobs=5,
                frequency_penalty=0,
                presence_penalty=0
            )
            break
        except:
            # openai.api_key = os.getenv(random.choice(keys))
            pass
    if cnt >= 100:
        return "none", "none", "No", 0.0
    time.sleep(1)
    return parse_gpt_response(response)


def parse_gpt_response(GPT3_response):
    if not GPT3_response:
        return "none", "none", "No", 0.0
    GPT3_response_txt = GPT3_response["choices"][0]["text"]
    GPT3_response_txt_lst = [x for x in GPT3_response_txt.split("\n") if len(x) > 0]
    if len(GPT3_response_txt_lst) == 0:
        return "none", "none", "No", 0.0
    norms = GPT3_response_txt_lst[0].split(';')

    if len(norms) >= 3:
        if 'No' in GPT3_response_txt:
            GPT3_response_txt_lst = [norms[0], GPT3_response_txt_lst[len(norms) - 1], 'No']
        elif "Yes" in GPT3_response_txt:
            GPT3_response_txt_lst = [norms[0], GPT3_response_txt_lst[len(norms) - 1], 'Yes']

    if len(GPT3_response_txt_lst) < 3:
        return "none", "none", "No", 0.0
    # violation_flag = True if "Yes" in GPT3_response_txt.lstrip().split() else False
    # violation_prob = 0.0
    try:
        YesNoIdx = GPT3_response["choices"][0]["logprobs"]['tokens'].index("Yes") if \
            "Yes" in GPT3_response_txt else GPT3_response["choices"][0]["logprobs"]['tokens'].index("No")
        yes_prob = GPT3_response["choices"][0]["logprobs"]['top_logprobs'][YesNoIdx]["Yes"]
        no_prob = GPT3_response["choices"][0]["logprobs"]['top_logprobs'][YesNoIdx]["No"]
        yes_prob, no_prob = math.exp(yes_prob), math.exp(no_prob)
        violation_prob = yes_prob / (yes_prob + no_prob)
    except:
        violation_prob = 0.85
    if ';' in GPT3_response_txt_lst[0][:-1]:
        return GPT3_response_txt_lst[0][:-1], GPT3_response_txt_lst[1], GPT3_response_txt_lst[2], violation_prob
    else:
        return GPT3_response_txt_lst[0], GPT3_response_txt_lst[1], GPT3_response_txt_lst[2], violation_prob

processed_audio = CCU.socket(CCU.queues['RESULT'], zmq.SUB)
result = CCU.socket(CCU.queues['RESULT'], zmq.PUB)
processed_messages = set()
norm_list = ['apology', 'greeting', 'criticism', 'request', 'persuasion']
norm_id_mapping = {'apology': '101', 'criticism': '102', 'greeting': '103', 'request': '104', 'persuasion': '105'}

flag = True

while True:
    if flag:
        logging.info("INTO WHILE LOOP...")
        flag = False

    # Start out by waiting for a message on the gesture queue.
    in_bound_message = CCU.recv(processed_audio)
    
    # Once we get that message, see if it is a pointing message, since
    # that is the only thing we care about.
    if in_bound_message and in_bound_message['type'] == 'asr_result':
        # In a real analytic we would now run the analysis to see if this
        # is an offensive action, but for this sample, we just assume that
        # all pointing is offensive.
        if not processed_messages or in_bound_message['uuid'] not in processed_messages:
            processed_messages.add(in_bound_message['uuid'])
            res_ = detect_norm(in_bound_message)
            norm_category_name = res_[0].split(';')[0] if ';' in res_[0] else res_[0]

            if norm_category_name in norm_list:
                # Creates a simple messsage of type 'offensive' with the detail
                # 'pointing'.  The base_message function also populates the 'uuid'
                # and 'datetime' fields in the message (which is really just a
                # Python dictionary).
                out_bound_message = CCU.base_message('norm_occurrence')
                out_bound_message['name'] = norm_id_mapping[norm_category_name] if norm_category_name in norm_id_mapping else 'none'
                logging.info('norm name is {}'.format(out_bound_message['name']))
                # out_bound_message['sentence'] = res_[1]
                out_bound_message['status'] = 'violate' if res_[2] == 'Yes' else 'adhere'
                out_bound_message['llr'] = res_[3]
                out_bound_message['trigger_id'] = [in_bound_message['uuid']]
                # Next we send it.
                CCU.send(result, out_bound_message)
                logging.info(out_bound_message)