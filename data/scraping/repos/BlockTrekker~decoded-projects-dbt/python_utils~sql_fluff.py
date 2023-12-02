import os
import openai
import json


# Go up two directories and read the API key
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'keys', 'openai_key.json'), 'r') as f:
    data = json.load(f)
    openai.api_key = data['key']

# Define the function to convert SQL to Big Query compatible version

def convert_sql_to_bigquery(sql):
    tokens = round(len(sql) / 3)
    max_tokens = 4000 - tokens
    response = openai.Completion.create(
      model="text-davinci-003",  # updated engine ID
      prompt=sql,
      temperature=0,
      max_tokens= max_tokens,
    )
    print(response.choices[0].text.strip())
    return response.choices[0].text.strip()


# Traverse the directory and subdirectories to find all files ending in .sql
# for subdir, dirs, files in os.walk('/home/outsider_analytics/Code/spellbook-may/models/balancer/arbitrum'):
#     count = 0
#     for file in files:
#         count = count + 1
#         if file.endswith('.sql'):
#             # Read the SQL query from the file
#             with open(os.path.join(subdir, file), 'r') as f:
#                 sql = f.read()
#             # Convert the SQL query to a Big Query compatible version
#             bigquery_sql = convert_sql_to_bigquery(f"make this query big query compatable without changing the jinja since its dbt: {sql}")
#             # Write the Big Query compatible version to the file
#             with open(os.path.join(subdir, file), 'w') as f:
#                 f.write(bigquery_sql)
#             print(f'Converted {file} to Big Query compatible version')
#             if count == 2:
#                 break
query = """
SELECT
    *
FROM 
    example.table
UNION
SELECT
    *
FROM 
    example.table2
"""

messages = [
    {"role": "system", "content": "You are a code-writing assistant."},
    {"role": "user", "content": "Make this sql query compatible with Big Query without changing any of the jinja. Do not provide any explanation or introduction, only the code."},
    {"role": "user", "content": f"{query}"},
]

max_tokens = 4000
tokens_generated = 0
responses = []

while tokens_generated < max_tokens:
    # Calculate the number of tokens used so far in the messages
    tokens_used = sum([len(msg['content'].split()) for msg in messages])

    print(f"Message tokens used: {tokens_used}")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Max tokens: {max_tokens}")
    print(f"Tokens left: {max_tokens-tokens_generated}")
    print(f"Messages: {messages}")
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=4000-tokens_used
    )
    
    # Append the response
    responses.append(response['choices'][0]['message']['content'])
    print("response:")
    print(response['choices'][0]['message']['content'])
    # Update tokens_generated
    tokens_generated += len(response['choices'][0]['message']['content'].split())

    # Add the assistant's message to the messages list to continue the conversation
    messages.append({"role": "assistant", "content": response['choices'][0]['message']['content']})

# Concatenate all responses
full_text = ' '.join(responses)

print(full_text)
