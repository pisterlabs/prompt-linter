from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import json
import argparse
from tqdm import tqdm
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
client = Anthropic(
    api_key=config["Anthropic"]["api_key"],
    timeout=60.0
)

def query(prompt: str) -> str:
    response = client.completions.create(
        model="claude-2",
        prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
        max_tokens_to_sample=600,
        temperature=0.0,
    )
    return response.completion

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_jsonl", 
        type=str, 
        required=True,
        help="Path to the jsonl that store generated formated prompts."
    )
    parser.add_argument(
        "--output_jsonl", 
        type=str, 
        required=False, 
        help="Path to the output jsonl for ChatGPT response."
    )

    args = parser.parse_args()
    output = []
    with open(args.prompt_jsonl, "r", encoding="UTF-8") as f:
        input_jsonl = f.readlines()
        for line in tqdm(input_jsonl):
            question = json.loads(line)
            prompt = question["prompt"]
            try:
                response = query(prompt)
            except Exception as e:
                print(e)
                response = "ERROR"
            prediction = {
                "id" : question["id"],
                "prompt": prompt,
                "full_response": response,
                "gold_answer": question["answer"]
            }
            output.append(prediction)
    
    with open(args.output_jsonl, "w", encoding="UTF-8") as f:
        for prediction in output:
            line = json.dumps(prediction, ensure_ascii=False)
            f.write(line+"\n")

    
    