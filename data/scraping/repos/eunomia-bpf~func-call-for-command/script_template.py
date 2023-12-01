def gen_bash_code(func_descript):
    pre_content = """#!/bin/bash
user_input="$@"
response=$(curl -s https://api.openai.com/v1/chat/completions -u :$OPENAI_API_KEY -H 'Content-Type: application/json' -d '{
  "model": "gpt-3.5-turbo-0613",
  "messages": [
    {"role": "user", "content": "'"$user_input"'"}
  ],
  "functions": [
"""
    post_content="""
]}')

# Parsing JSON data
full_command=$(echo "$response" | jq -r '.choices[0].message.function_call.name')
args=$(echo "$response" | jq '.choices[0].message.function_call.arguments')

args=$(echo -e $args | tr -d '\\\\')
args=$(echo $args | sed 's/^"//;s/"$//')

for key in $(echo "$args" | jq -r 'keys[]'); do
    value=$(echo $args | jq -r --arg key $key '.[$key]')
    if [ "$value" != "true" ] && [ "$value" != "false" ]; then
        full_command+=" --$key "$value" "
    else
        full_command+=" --$key "
    fi
done

echo "Run: $full_command"
eval "$full_command"
"""
    return pre_content + func_descript + post_content

def gen_python_code(func_descript):
    pre_content="""#!/bin/python
import subprocess
import openai
import json
import sys
user_input = " ".join(sys.argv[1:])

# Send the conversation and available functions to GPT
messages = [{"role": "user", "content": user_input}]
functions = ["""
    post_content = """]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
    functions=functions,
    function_call="auto",  # auto is default, but we'll be explicit
)
response_message = response["choices"][0]["message"]

# Check if GPT wanted to call a function
if response_message.get("function_call"):
    full_command = []
    full_command.append(response_message["function_call"]["name"])
    args = json.loads(response_message["function_call"]["arguments"])

    for key, value in args.items():
        if (value is not True) and (value is not False):
            full_command.extend([f"--{key}",  f'"{value}"'])
        else:
            full_command.append(f"--{key}")
    print("Run: ", " ".join(full_command))
    subprocess.run(full_command, text=True)
"""
    return pre_content + func_descript + post_content

