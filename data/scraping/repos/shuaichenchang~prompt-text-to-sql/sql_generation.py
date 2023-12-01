import os
import time
import openai

import asyncio
from utils import format_query, get_prompt_length

DB_SEP = "/**/\n"
BATCH_SIZE = 30
MAX_GEN_TOKENS = 200


async def dispatch_openai_requests(
        messages_list,
        model,
        temperature,
        max_tokens,
        top_p,
        stop,
):
    async def call_openai(message):
        while True:
            try:
                response = await openai.ChatCompletion.acreate(
                    model=model,
                    messages=message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=stop,
                )
                return response
            except Exception as e:
                print(e)
                await asyncio.sleep(30)

    async_responses = [
        call_openai(message) for message in messages_list
    ]
    return await asyncio.gather(*async_responses)


def cut_prompt_with_max_tokens(openai_model, prompt, max_generate_tokens=MAX_GEN_TOKENS, setting="crossdomain"):
    if openai_model == "codex":
        model_max_tokens = 8000
    else:
        model_max_tokens = 4000
    prompt_len = get_prompt_length(prompt, model=openai_model)
    cnt = 0
    while prompt_len >= model_max_tokens - max_generate_tokens:
        prompt = prompt.split(DB_SEP)
        prompt = DB_SEP.join([""] + prompt[2:])
        prompt_len = get_prompt_length(prompt, model=openai_model)
        cnt += 1
    if cnt > 0:
        print(f"Prompt too long, skip the first {cnt} databases.")
        if setting != "crossdomain":
            raise Exception("Cannot skip databases for this setting.")
    return prompt, prompt_len


def call_codeX(openai_model, prompt, max_tokens=MAX_GEN_TOKENS, stop=[";", "Question", 'Answer', '/*'], num_return=1, temperature=0, top_p=1):
    if openai_model == "codex":
        model = "code-davinci-002"
    else:
        raise NotImplementedError

    prompt_len = get_prompt_length(prompt, model=openai_model)

    while (True):
        try:
            response = openai.Completion.create(
                model=model,
                prompt=prompt,
                n=num_return,
                best_of=num_return,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop,
                logprobs=5
            )
            break
        except Exception as e:
            print(e, "Retry.")
            time.sleep(10)
            continue

    for i in range(len(response["choices"])):
        response["choices"][i]["text"] = response["choices"][i]["text"].replace('\n', ' ').replace('    ', ' ').replace('\t', ' ')
    return response, prompt_len


def text_to_sql_direct(openai_model, questions, prompt_template, demo_sql_format="normalized"):
    if demo_sql_format == "normalized":
        select = "select"
    elif demo_sql_format == "unnormalized":
        select = "SELECT"
    else:
        raise NotImplementedError
    predictions = []
    prompts = []
    prompts_len = []
    stop = [";", "Question", 'Answer', '/*']
    for q_id, q in enumerate(questions):
        prompt = prompt_template + f"Question: {q['question']}\n" + select
        prompt, prompt_len = cut_prompt_with_max_tokens(openai_model, prompt, MAX_GEN_TOKENS, setting="zeroshot")
        prompts.append(prompt)
        prompts_len.append(prompt_len)

    if openai_model == "chatgpt":  # batch call ChatGPT to speed up
        responses = []
        for i in range(0, int((len(prompts) + BATCH_SIZE - 1) / BATCH_SIZE)):
            responses_batch = asyncio.run(
                dispatch_openai_requests(
                    messages_list=[[{"role": "user", "content": prompt}] for prompt in prompts[i * BATCH_SIZE:min(len(prompts), (i + 1) * BATCH_SIZE)]],
                    model="gpt-3.5-turbo-0301",
                    temperature=0,
                    max_tokens=MAX_GEN_TOKENS,
                    top_p=1.0,
                    stop=stop,
                )
            )
            responses += responses_batch
            time.sleep(10)
        for q, response in zip(questions, responses):
            x = response["choices"][0]["message"]["content"].replace('\n', ' ').replace('    ', ' ').replace('\t', ' ')
            response["choices"][0]["text"] = ' ' + x
            sql = select + response["choices"][0]["text"]
            print(q["question"])
            print(sql)
    elif openai_model == "codex":
        responses = []
        for q, prompt in zip(questions, prompts):
            response, prompt_len = call_codeX(openai_model, prompt, max_tokens=MAX_GEN_TOKENS, stop=stop)
            responses.append(response)
            sql = select + response["choices"][0]["text"]
            print(q["question"])
            print(sql)

    else:
        raise NotImplementedError
    for q_id, (q, response, prompt_len) in enumerate(zip(questions, responses, prompts_len)):
        sql = select + response["choices"][0]["text"]
        predictions.append({
            "db_id": q["db_id"],
            "question": q["question"],
            "gold_sql": q["query"],
            "predicted_sql": sql,
            "prompt_len": prompt_len,
        })
    return predictions, prompts


def text_to_sql_few_shot_singledomain(openai_model, questions, indomain_schema, indomain_demo_examples_per_question, demo_sql_format="normalized"):
    if demo_sql_format == "normalized":
        select = "select"
    elif demo_sql_format == "unnormalized":
        select = "SELECT"
    else:
        raise NotImplementedError
    print("=" * 10 + "start" + "=" * 10)
    few_shot_in_prompts = []
    predictions = []
    prompts = []
    prompts_len = []

    for q_id, (q, indomain_few_shot_examples) in enumerate(zip(questions, indomain_demo_examples_per_question)):
        prompt = indomain_schema
        indomain_demonstration = []
        for example in indomain_few_shot_examples:
            prompt += f"Question: {example['question']}\n"
            query = format_query(example, demo_sql_format)
            prompt += query + '\n'
            indomain_demonstration.append([example["question"], query])
        few_shot_in_prompts.append([q["question"], q["query"], indomain_demonstration])

        prompt += f"Question: {q['question']}\n" + select
        prompt, prompt_len = cut_prompt_with_max_tokens(openai_model, prompt, MAX_GEN_TOKENS, setting="singledomain")
        prompts_len.append(prompt_len)
        prompts.append(prompt)

    stop = [";", "Question", 'Answer', '/*']
    if openai_model == "chatgpt":  # batch call ChatGPT to speed up
        responses = []
        for i in range(0, int((len(prompts) + BATCH_SIZE - 1) / BATCH_SIZE)):
            responses_batch = asyncio.run(
                dispatch_openai_requests(
                    messages_list=[[{"role": "user", "content": prompt}] for prompt in prompts[i * BATCH_SIZE:min(len(prompts), (i + 1) * BATCH_SIZE)]],
                    model="gpt-3.5-turbo-0301",
                    temperature=0,
                    max_tokens=MAX_GEN_TOKENS,
                    top_p=1.0,
                    stop=stop,
                )
            )
            responses += responses_batch
            time.sleep(10)
        for q, response in zip(questions, responses):
            x = response["choices"][0]["message"]["content"].replace('\n', ' ').replace('    ', ' ').replace('\t', ' ')
            response["choices"][0]["text"] = ' ' + x
            sql = select + response["choices"][0]["text"]
            print(q["question"])
            print(sql)

    elif openai_model == "codex":
        responses = []
        for q, prompt in zip(questions, prompts):
            response, prompt_len = call_codeX(openai_model, prompt, max_tokens=MAX_GEN_TOKENS, stop=stop)
            responses.append(response)
            sql = select + response["choices"][0]["text"]
            print(q["question"])
            print(sql)
    else:
        raise NotImplementedError
    for q_id, (q, response, prompt_len) in enumerate(zip(questions, responses, prompts_len)):
        sql = select + response["choices"][0]["text"]
        predictions.append({
            "db_id": q["db_id"],
            "question": q["question"],
            "gold_sql": q["query"],
            "predicted_sql": sql,
            "prompt_len": prompt_len,
        })
    return few_shot_in_prompts, predictions


def create_outdomain_prompt(outdomain_schemas, outdomain_demo_examples, demo_sql_format="normalized"):
    prompt = ""
    outdomain_demostration = []
    for schema, examples in zip(outdomain_schemas, outdomain_demo_examples):
        prompt += DB_SEP
        prompt += schema
        outdomain_demostration.append([])
        for example in examples:
            prompt += f"Question: {example['question']}\n"
            query = format_query(example, demo_sql_format)
            prompt += query + '\n'
            outdomain_demostration[-1].append([example["question"], query])
        prompt += '\n'
    return prompt, outdomain_demostration


def text_to_sql_few_shot_crossdomain(openai_model, questions, outdomain_schemas_per_question, indomain_schema, outdomain_demo_examples_per_question,
                                     demo_sql_format="normalized"):
    if demo_sql_format == "normalized":
        select = "select"
    elif demo_sql_format == "unnormalized":
        select = "SELECT"
    else:
        raise NotImplementedError

    print("=" * 10 + "start" + "=" * 10)
    few_shot_in_prompts = []
    predictions = []
    for q_id, (q, outdomain_schemas, outdomain_demo_examples) in enumerate(
            zip(questions, outdomain_schemas_per_question, outdomain_demo_examples_per_question)):
        prompt, outdomain_demostration = create_outdomain_prompt(outdomain_schemas, outdomain_demo_examples, demo_sql_format=demo_sql_format)
        prompt += DB_SEP
        prompt += indomain_schema
        few_shot_in_prompts.append([q["question"], q["query"], outdomain_demostration])

        prompt += f"Question: {q['question']}\n" + select
        prompt, prompt_len = cut_prompt_with_max_tokens(openai_model, prompt, MAX_GEN_TOKENS, setting="crossdomain")
        response, prompt_len = call_codeX(openai_model=openai_model, prompt=prompt)
        sql = select + response["choices"][0]["text"]
        print(q["question"])
        print(sql)
        predictions.append({
            "db_id": q["db_id"],
            "question": q["question"],
            "gold_sql": q["query"],
            "predicted_sql": sql,
            "prompt_len": prompt_len,
        })
    return few_shot_in_prompts, predictions
