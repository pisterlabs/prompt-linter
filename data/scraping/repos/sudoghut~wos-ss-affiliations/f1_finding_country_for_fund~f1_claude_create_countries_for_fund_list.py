import pandas as pd
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import os


input_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fund_list_small.csv")
prompt_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "prompt.txt")
output_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "claude_output")
max_tokens_to_sample_value = 10000
input_list = pd.read_csv(input_file, dtype=str).fillna("").values.tolist()
input_list = [item[0] for item in input_list]

with open('api_key.txt', 'r') as file:
    api_key_str = file.read()

anthropic = Anthropic(
    api_key = api_key_str,
)

for filename in os.listdir(output_folder):
    file_path = os.path.join(output_folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

with open(os.path.join(prompt_file), "r") as file:
    prompt_str = file.read()
prompt_str = prompt_str.replace("\\n", "\n")

print(prompt_str)
print(input_list[:5])
print(len(input_list))

count = 0
batch = 150
batch_list = []
for i in range(len(input_list)):
    batch_list.append(input_list[i])
    if i % batch == 0 and i != 0:
        output_names = "\n".join(batch_list)
        prompt = f"{prompt_str}{output_names}"
        completion = anthropic.completions.create(
            model="claude-1",
            max_tokens_to_sample=max_tokens_to_sample_value,
            prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
        )
        file = open(output_folder + "\\output_" + str(count) + ".txt", "w", encoding="utf-8")
        file.write(completion.completion)
        file.close()
        batch_list = []
    count += 1
if len(batch_list) != 0:
    count += 1
    output_names = "\n".join(batch_list)
    prompt = f"{prompt_str}{output_names}"
    completion = anthropic.completions.create(
        model="claude-1",
        max_tokens_to_sample=max_tokens_to_sample_value,
        prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
    )
    file = open(output_folder + "\\output_" + str(count) + ".txt", "w", encoding="utf-8")
    file.write(completion.completion)

print("Finished!")