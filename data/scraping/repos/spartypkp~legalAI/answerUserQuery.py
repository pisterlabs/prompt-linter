import openai
import utilityFunctions as util
import config
import promptStorage as prompts
import time
import asyncio
import json
import app

def main():
    pass
    # 15047 tokens
    #response_list = separate_answer(question_list[2], True, lawful[0:12], "gpt-3.5-turbo")
    #print(response_list)
    

def answering_stage(question_list, legal_text, user_query):
    print("Starting answering stage...")
    responses_list = separate_answer(question_list[2], legal_text[2], "gpt-3.5-turbo-16k")
    
    begin = time.time()
    print("  - Creating answer template with GPT 4")
    summaryTemplate = create_summary_template(question_list[2], responses_list)
   

    end = time.time()
    print("    * Total time: {}".format(round(end-begin, 2)))
    return summaryTemplate, responses_list, question_list[2]
    


def separate_answer(question, legal_text, model):
    message_list = []
    response_list = []
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    for section in legal_text:

        message_list.append(prompts.get_prompt_simple_answer(section, question))
    
    begin = time.time()
    
    results = asyncio.run(util.get_completion_list(message_list, 100, used_model=model))
    
    for completion in results:
        #print(completion)
        #prompt_tokens += completion["usage"]["prompt_tokens"]
        #completion_tokens += completion["usage"]["completion_tokens"]
        #total_tokens += completion["usage"]["total_tokens"]
        response_list.append(completion["choices"][0]["message"]["content"])
    #total_cost = util.calculate_prompt_cost(model, prompt_tokens, completion_tokens)
    
    response_str = ""
    for response in response_list:
        if "[IGNORE]" in response:
            continue
        total_tokens += util.num_tokens_from_string(response)
        response_str = response_str + "====\n" + response + "\n"
    
    end = time.time()
    #print("    * Total time: {}, Total Tokens: {}, Total Cost: ${}".format(round(end-begin, 2), total_tokens, round(total_cost, 2)))
        
    return response_str
    #print(response_list)

def create_summary_template(question, legal_documentation):
    prompt_summarize = prompts.get_prompt_summary_template(question, legal_documentation)
    chat_completion = util.create_chat_completion(used_model="gpt-4", prompt_messages=prompt_summarize, temp=1, api_key_choice="will", debug_print=True)
    result_str = chat_completion.choices[0].message.content
    return result_str
    
def populate_summary_template(question, legal_documentation, template):
    prompt_populate = prompts.get_prompt_populate_summary_template(question, template, legal_documentation)
    chat_completion = util.create_chat_completion(used_model="gpt-3.5-turbo-16k", prompt_messages=prompt_populate, temp=0, api_key_choice="will", debug_print=True)
    result_str = chat_completion.choices[0].message.content
    return result_str


def answer_one_question(prompt_final_answer, use_gpt_4):
    model = "gpt-3.5-turbo-16k"
    if use_gpt_4:
        model="gpt-4"
    who = "will"
    

    chat_completion = util.create_chat_completion(used_model=model, prompt_messages=prompt_final_answer, temp=0.2, api_key_choice=who)
    result_str = chat_completion.choices[0].message.content

    prompt_tokens = chat_completion.usage["prompt_tokens"]
    completion_tokens = chat_completion.usage["completion_tokens"]
    cost = util.calculate_prompt_cost(model, prompt_tokens, completion_tokens)
    return result_str, prompt_tokens, completion_tokens, cost

if __name__ == "__main__":
    main()