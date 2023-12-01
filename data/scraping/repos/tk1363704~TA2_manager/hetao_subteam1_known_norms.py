import os
import time
import math
import openai
import random
def in_context_gpt3(input_text, prompt=''):
    '''input: a couple of sentences.
       output: None means no norm is detected. Or return 4 teams:
       1. Norm category: e.g. applogy
       2. utterances contain the norm
       3. Yes or No.  means wheather or not violate the norm
       4. The probability of violation. A smaller number means less probability of violation.

       Example of input: 晓明，早上不好意思，我不是故意的。 没事~
       Example of output: 'apology', '"晓明，早上不好意思，我不是故意的"', 'No', 0.0271
       Note that the output of 'No norm' means no norm is detected in the utterances.
    '''
    keys=["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","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",
          "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",
          "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",
          "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"]
    # openai.api_key = os.getenv(random.choice(keys))
    cnt = 0
    while cnt < 100:
        cnt += 1
        try:
            response = openai.Completion.create(
                model="text-davinci-002",
                api_key=random.choice(keys),
                prompt="Given the dialogue:\n"
                       "你穿哪个衣服好难看，还是放弃吧。 你为啥这么说，我很伤心\n"
                       "Which social norm does the above dialogue contain?  Which does utterance show the social norm? Is there any social norm violation?\n"
                       "none; apology; greeting; criticism; request; persuasion;"
                       "\n"
                       "criticism;\n\"你穿哪个衣服好难看，还是放弃吧\"\nYes\n"
                       "Given the dialogue:\n"
                       + input_text+
                       "\nWhich social norm does the above dialogue contain?  Which does utterance show the social norm? Is there any social norm violation?\n"
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
            pass
    if cnt >=100:
        return "No norm"
    time.sleep(1)
    return parse_gpt_response(response)
def parse_gpt_response(GPT3_response):
    GPT3_response_txt=GPT3_response["choices"][0]["text"]
    GPT3_response_txt_lst = [x for x in GPT3_response_txt.split("\n") if len(x)>0]
    if len(GPT3_response_txt_lst) < 3:
        return "No norm"
    try:
        YesNoIdx = GPT3_response["choices"][0]["logprobs"]['tokens'].index("Yes") if \
            "Yes" in GPT3_response_txt else GPT3_response["choices"][0]["logprobs"]['tokens'].index("No")
        yes_prob = GPT3_response["choices"][0]["logprobs"]['top_logprobs'][YesNoIdx]["Yes"]
        no_prob = GPT3_response["choices"][0]["logprobs"]['top_logprobs'][YesNoIdx]["No"]
        yes_prob, no_prob = math.exp(yes_prob), math.exp(no_prob)
        violation_prob = yes_prob / (yes_prob+no_prob)
    except:
        violation_prob = 0.85
    return GPT3_response_txt_lst[0][:-1], GPT3_response_txt_lst[1], GPT3_response_txt_lst[2], violation_prob

if __name__ =='__main__':

    res = in_context_gpt3("晓明，早上不好意思，我不是故意的。 没事~")
    print(res) # 'apology', '"晓明，早上不好意思，我不是故意的"', 'No', 0.0271



