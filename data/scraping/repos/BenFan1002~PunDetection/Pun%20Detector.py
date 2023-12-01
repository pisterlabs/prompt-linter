import openai
from Key import api_key
lines = []

print("Enter your input. Press Enter on an empty line to finish:")

while True:
    line = input()

    if line:
        lines.append(line)
    else:
        break

text = "\n".join(lines)
openai.api_key = api_key
print("You entered:", text)
# prompt for pun detection and output in the form of json
prompt = f"""
Does the text contain a pun:
text: \"{text}\"

output in the form of json:
{{
'is pun': True or False,
# If the text contains a pun, the following fields will be present
'Explanation 1': ,
'Explanation 2': ,
}}
"""
# Check if lecture chunk is in cache and use cached result if available
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)
lecture_chunk = response.choices[0].message.content
response = eval(lecture_chunk)
if response['is pun']:
    print("The text contains a pun.")
    print("Explanation 1:", response['Explanation 1'])
    print("Explanation 2:", response['Explanation 2'])
else:
    print("The text does not contain a pun.")
