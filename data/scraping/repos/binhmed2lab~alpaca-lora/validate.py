import fire
from datetime import datetime

from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import sys
from transformers import GenerationConfig
import json
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import copy
from utils import openai_utils
from finetune import (SYS_POSTFIX, 
                      SYS_PREFIX, 
                      INST_POSTFIX, 
                      INST_PREFIX, 
                      OUTPUT_POSTFIX, 
                      OUTPUT_PREFIX,
                      preprocess,
                      generate_response,
                      ask_alpaca,
                      read_json)

def write_json(obj, path):
    if not path.endswith(".json"):
        path += ".json"

    json_object = json.dumps(obj, indent=4, ensure_ascii=False)
    with open(path, "w", encoding="utf-8") as outfile:
        outfile.write(json_object)

def batch_inference(data, model, tokenizer, batch_size = 4, max_length = 1500):
    tk = tqdm(range(0, len(data), batch_size))
    predictions = []
    for start_idx in tk:
        batch = data[start_idx:start_idx+batch_size]
        preds = ask_alpaca(batch, model, tokenizer, max_length = max_length)
        predictions += preds
        examples = [p[:50] for p in preds]
        tk.set_postfix(
            examples=examples,
        )
    return predictions

def get_dialog_string(dialog):
    prompt = ""
    roles = [msg["role"] for msg in dialog]
    messages = [msg["content"] for msg in dialog]

    if roles[0].upper() == "SYSTEM":
        prompt += f"{SYS_PREFIX}{messages[0]}{SYS_POSTFIX}"

    for role, msg in zip(roles, messages):
        if role.upper() == "ASSISTANT":
            prompt += f"{msg}{OUTPUT_POSTFIX}"
        elif role.upper() == "USER":
            prompt += f"{INST_PREFIX}{msg}{INST_POSTFIX}{OUTPUT_PREFIX}"
    
    return prompt

def ValidateFinetunePerformance(model, tokenizer, data, data_name, gpt_model, batch_size = 6, max_length = 1500, test_limit = -1):
    if isinstance(test_limit, int) and test_limit > -1:
        data = data[:test_limit]

    t = len(data)
    dialog_strings = [get_dialog_string(d['dialog']) for d in data]

    dialogs_ids = tokenizer(dialog_strings)['input_ids']
    data = [data[idx] for idx, d in enumerate(dialogs_ids) if len(d) < max_length]
    tt = len(data)
    print(f"Remove {t-tt} items > {max_length} lengths. Has {tt} items left")

    print("Start validating:", data_name)
    predict_items = []

    for item_id, data_point in enumerate(data):
        dialog = data_point['dialog']
        prompt = ""

        roles = [msg["role"] for msg in dialog]
        messages = [msg["content"] for msg in dialog]

        if roles[0].upper() == "SYSTEM":
            prompt += f"{SYS_PREFIX}{messages[0]}{SYS_POSTFIX}"

        prev_role = roles[0].upper()

        for dialog_pos, (role, msg) in enumerate(zip(roles, messages)):
            if role.upper() == "ASSISTANT":
                if prev_role == "USER":
                    predict_items.append({
                        "prompt": prompt,
                        "answer": msg,
                        "item_id": item_id,
                        "dialog_position": dialog_pos
                    })
                prompt += f"{msg}{OUTPUT_POSTFIX}"
            elif role.upper() == "USER":
                prompt += f"{INST_PREFIX}{msg}{INST_POSTFIX}{OUTPUT_PREFIX}"

            prev_role = role.upper()


    prompts = [p['prompt'] for p in predict_items]
    results = batch_inference(prompts, model, tokenizer, batch_size = batch_size, max_length = max_length)

    print("Start prediction")
    for result, predict_item in zip(results, predict_items):
        item_id = predict_item['item_id']
        dialog_position = predict_item['dialog_position']
        predict_dialog = data[item_id].get('predict_dialog')
        if predict_dialog is None:
            data[item_id]['predict_dialog'] = copy.deepcopy(data[item_id]['dialog'])
            predict_dialog = data[item_id]['predict_dialog']

        predict_dialog[dialog_position]['content'] = result

    print(f"Start Validate by {gpt_model}")

    tk = tqdm(data, total=len(data))
    answer_choices = ["Assistant A", "Assistant B", "Equally Good", "Equally Bad"]
    stats = {a: 0 for a in answer_choices}
    total_usage = {
        "completion_tokens": 0,
        "prompt_tokens": 0,
    }

    for d in tk:
        gpt4_dialog = d['dialog']
        llama2_dialog = d['predict_dialog']
        package = openai_utils.Compare2Dialog(
            dialog_a=gpt4_dialog,
            dialog_b=llama2_dialog,
            answer_choices=answer_choices,
            model=gpt_model
        )
        analyzed_result = package['response']
        usage = package['usage']
        total_usage['completion_tokens'] += usage['completion_tokens']
        total_usage['prompt_tokens'] += usage['prompt_tokens']

        d['analyzed_result'] = analyzed_result
        if analyzed_result in stats:
            stats[analyzed_result] += 1
            tk.set_postfix(
                stats = stats
            )
        else:
            tk.set_postfix(
                Error = analyzed_result
            )
    
    gpt_pricing = {
        "gpt-3.5-turbo": (0.002, 0.002),
        "gpt-3.5-turbo-0613": (0.002, 0.002),
        "gpt-4": (0.03, 0.06),
        "text-davinci-003": (0.02, 0.02)
    }
    prompt_p, complete_p = gpt_pricing[gpt_model]
    prompt_cost = total_usage['prompt_tokens'] * prompt_p / 1000
    completion_cost = total_usage['completion_tokens'] * complete_p / 1000
    final_cost = prompt_cost + completion_cost
    print(f"Experiment cost: prompt {prompt_cost} | completion {completion_cost} | final {final_cost}")
    print(stats)
    today = datetime.now()
    finalize = {
        "run_data": data,
        "result": stats,
        "chatgpt_cost": {
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "final_cost": final_cost
        }
    }
    write_json(finalize, f"{data_name}_{today.day}_{today.month}_{today.year}")
    return data


def validate(
    data_path: str,
    data_name: str,
    lora_model: str,
    max_length: int = 1500,
    batch_size: int = 4,
    openaikey: str = None,
    gpt_model: str = "gpt-3.5-turbo",
    test_limit: int = -1
):
    import openai
    openai.api_key = openaikey

    validate_data = read_json(data_path)
    device_map = "auto"
    config = PeftConfig.from_pretrained(lora_model)
    base_model = config.base_model_name_or_path

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    model = PeftModel.from_pretrained(model, lora_model)
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    
    ValidateFinetunePerformance(
        model=model,
        tokenizer=tokenizer,
        data=validate_data,
        data_name=data_name,
        batch_size=batch_size,
        max_length = max_length,
        gpt_model = gpt_model,
        test_limit = test_limit
    )
    

if __name__ == "__main__":
    fire.Fire(validate)

   

