"""
Evaluate the selected instruction on Llama-2
This script needs accelerate==0.21 and transformers==4.31

Run command:
accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    optimization/evaluation/evaluate_selected_instruction_llama2.py \
    SCRIPT_CMD_ARGS
"""
import json
import os
import pdb
import sys
import math
import random
import shutil
import openai
import argparse
from tqdm import tqdm
from datetime import datetime
from rouge_score import rouge_scorer
sys.path.append('.')
import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from optimization.Constants import NIV2_TEST_TASK_FORMAT_INSTRUCTION
RougeScorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)


def read_jsonl_as_list(path: str):
    assert path.endswith('.jsonl')
    with open(path, 'r', encoding='utf8') as fin:
        result = []
        for line in fin:
            data = json.loads(line.strip())
            result.append(data)
    # print(f'Read {len(result)} data from {path}')
    return result


def save_list_as_jsonl(path: str, data):
    assert path.endswith('.jsonl')
    with open(path, 'w', encoding='utf8') as fout:
        for instance in data:
            fout.write(json.dumps(instance))
            fout.write('\n')
    print(f'Saved {len(data)} data to {path}')


def mean(array):
    assert isinstance(array, list)
    return sum(array) / len(array)


few_shot_prompt = """{{instruction}}

{{demo}}

Input: {{input}}
Output:"""

zero_shot_prompt = """{{instruction}}

Input: {{input}}
Output:"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-prediction', type=str, default='CKPT_NAME/test_predictions.json',
                        help='Path to the model prediction file')
    parser.add_argument('-test_data', type=str,
                        default='DATA_DIR/test.jsonl')
    parser.add_argument('-output_file', type=str,
                        default='CKPT_NAME/llama2_test_predictions_inference_results.json')
    parser.add_argument('-model_name_or_path', type=str, default='meta-llama/Llama-2-13b-chat-hf',
                        help='Llama model name on HuggingFace')
    parser.add_argument('-precision', type=str, default='bf16', choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument('-max_input_tokens', type=int, default=2048,
                        help='Max input sequence length')
    parser.add_argument('-max_output_tokens', type=int, default=80,
                        help='Max output sequence length')
    parser.add_argument('-zero_shot', action='store_true', default=False,
                        help='If True, then conduct zero-shot inference')
    parser.add_argument('-random_baseline', action='store_true', default=False,
                        help='If True, randomly select an instruction from all candidate instructions '
                             'as the random instruction baseline')
    parser.add_argument('-naive_baseline', action='store_true', default=False,
                        help='If True, calculate the average score of the naive instruction')
    parser.add_argument('-append_format', action='store_true', default=False,
                        help='If True, append the formatting instruction to the end of the original instruction. '
                             'Usually used for zero-shot inference')
    args = parser.parse_args()
    assert not (args.random_baseline and args.naive_baseline)
    accelerator = Accelerator()

    if args.precision == "fp32":
        precision = torch.float32
    elif args.precision == "fp16":
        precision = torch.float16
    elif args.precision == "bf16":
        precision = torch.bfloat16
    else:
        raise ValueError("Unknown precision %s", args.precision)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", torch_dtype=precision)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    generation_config = GenerationConfig.from_pretrained(
        args.model_name_or_path,
        max_new_tokens=args.max_output_tokens,
        top_p=0.0,
        temperature=0.0,
        do_sample=False,
    )

    all_predictions = json.load(open(args.prediction, 'r', encoding='utf8'))
    test_data = read_jsonl_as_list(args.test_data)

    if os.path.exists(args.output_file):
        raise FileExistsError(f'Output file {args.output_file} already exists!')
    
    add_format_cnt = 0
    all_inference_examples = []

    for example in test_data:
        id_ = example['id']
        input_text = example['input']
        target_text = example['target']
        task_name = '_'.join(id_.split('_')[:-1])

        instruction_list = example['instructions']
        demos = example['demos']

        prediction = all_predictions[id_]
        pred_instruction_idx = prediction['select_instruction']
        pred_scores = prediction['predict_scores']

        assert len(pred_scores) == len(instruction_list)
        selected_instruction = instruction_list[pred_instruction_idx]

        # For the random baseline
        if args.random_baseline:
            selected_instruction = random.choice(instruction_list)
        elif args.naive_baseline:
            selected_instruction = instruction_list[0]

        # For zero-shot prompting, add the formatting instructions to avoid free-form generation
        if args.append_format and task_name in NIV2_TEST_TASK_FORMAT_INSTRUCTION.keys():
            # For random selection, if we happens to select the naive instruction,
            # we don't need to add additional formatting instructions since it originally has
            if selected_instruction != instruction_list[0]:
                selected_instruction += '\n' + NIV2_TEST_TASK_FORMAT_INSTRUCTION[task_name]
            add_format_cnt += 1

        prompt_template = zero_shot_prompt if args.zero_shot else few_shot_prompt
        prompt = prompt_template.replace('{{instruction}}', selected_instruction)
        prompt = prompt.replace('{{demo}}', demos)
        prompt = prompt.replace('{{input}}', input_text)
        
        all_inference_examples.append({
            "id": id_,
            "input": input_text,
            "target": target_text,
            "selected_instruction": selected_instruction,
            "prompt": prompt
        })

    print(f'Constructed {len(all_inference_examples)} examples for inference')
    print(f"[{datetime.now().strftime('%H:%M:%S')}] <GPU {accelerator.process_index}> Start generating...")

    my_outputs = []
    truncated_sequence_cnt = 0
    with accelerator.split_between_processes(all_inference_examples) as examples_curr_process:

        with torch.inference_mode():
            for example in tqdm(examples_curr_process, desc=f"GPU {accelerator.process_index}"):
                prompt = example['prompt']

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=args.max_input_tokens,
                    truncation=True,
                )
                input_ids = inputs.input_ids.to(model.device)
                attention_mask = inputs.attention_mask.to(model.device)

                if input_ids.size(1) == args.max_input_tokens:
                    truncated_sequence_cnt += 1

                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                )
        
                output = outputs[0]
                output_string = tokenizer.decode(
                    output[input_ids.size(1) :], skip_special_tokens=True
                )
                example['prediction'] = output_string.strip()
                target_text = example['target']
                example['score'] = RougeScorer.score(
                    target=target_text.strip(), 
                    prediction=output_string.strip()
                )['rougeL'].fmeasure

                my_outputs.append(example)

        output_path_curr_process = args.output_file + f".{accelerator.process_index}"
        json.dump(
            my_outputs,
            open(output_path_curr_process, "w", encoding="utf8"),
            indent=4,
            ensure_ascii=False,
        )

    print(f"[{datetime.now().strftime('%H:%M:%S')}] <GPU {accelerator.process_index}> Finished generation! "
          f"Truncated sequences: {truncated_sequence_cnt}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # concatenate outputs from all processes
        all_outputs = []
        for i in range(accelerator.num_processes):
            output_path_curr_process = args.output_file + f".{i}"
            all_outputs += json.load(
                open(output_path_curr_process, "r", encoding="utf-8")
            )
            os.remove(output_path_curr_process)

        all_outputs = sorted(all_outputs, key=lambda x: x["id"])
        json.dump(
            all_outputs,
            open(args.output_file, "w", encoding="utf8"),
            indent=4,
            ensure_ascii=False,
        )

        # collect scores for each task
        task2scores = {}
        for example in all_outputs:
            id_ = example['id']
            task_name = '_'.join(id_.split('_')[:-1])
            score = example['score']

            if task_name not in task2scores:
                task2scores[task_name] = []
            task2scores[task_name].append(score)

        # caclulate macro average score
        all_task_scores = []
        for task_name, score_list in task2scores.items():
            assert len(score_list) == 200
            all_task_scores.append(mean(score_list))
            print(f'{task_name}: ROUGE {mean(score_list):.2%}')

        print(f'Overall: ROUGE {mean(all_task_scores):.2%}')
        print(f'Examples where formatting instructions are added: {add_format_cnt}')


if __name__ == '__main__':
    main()
