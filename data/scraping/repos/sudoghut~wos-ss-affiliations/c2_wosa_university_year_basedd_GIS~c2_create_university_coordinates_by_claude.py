import pandas as pd
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import os

# input_excel = "node-with-country-small.xlsx"
input_excel = "node-with-country.xlsx"
output_folder = "claude_coordinates_output"
max_tokens_to_sample_value = 2000

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

input_df = pd.read_excel(input_excel, sheet_name="node-with-countries", dtype=str).fillna("")

print(input_df.head())
print(input_df.shape)

with open(os.path.join("c2_wosa_university_year_basedd_GIS","c2_clause_prompt.txt"), "r") as file:
    prompt_str = file.read()
prompt_str = prompt_str.replace("\\n", "\n")
print(prompt_str)

label_list = input_df["Label"].tolist()
count = 0
batch = 50
batch_list = []
for i in range(len(label_list)):
    batch_list.append(label_list[i])
    if i % batch == 0 and i != 0:
        output_names = "\n".join(batch_list)
        prompt = f"{prompt_str}{output_names}"
        completion = anthropic.completions.create(
            model="claude-1",
            # max_tokens_to_sample=300,
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