import json
import openai
import sys 

# Get the API key from the command line
api_key = arg1 = sys.argv[1]

# Input and output JSON file paths
input_file_path = 'input.json'
output_file_path = 'output.json'

def ChatGPTApi(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        api_key=api_key
    )
    return response.choices[0].message.content

# Load input data from the JSON file
with open(input_file_path, mode='r', encoding="utf-8") as input_file:
    input_data = json.load(input_file)


''' input data format
{
    problem_id: number,
    status: string,
    problem_statement: string,
    problem_input: string,
    problem_output:string
    problem_notes: string
    examples: [
      {
        "input": "",
        "output": ""
      }
    ]
},
'''
# Process each object in the list and update the 'problem_statement' field
for i, item in enumerate(input_data):
    print(f"Processing item {i+1} of {len(input_data)}")
    prompt = f"This question is a coding question: {item['problem_statement']}\n and will have the input of: {item['problem_input']}\n Rewrite the question to make it simpler for a computer to understand: "
    item['problem_statement'] = ChatGPTApi(prompt)

# Save the updated data to the output JSON file
with open(output_file_path, 'w') as output_file:
    json.dump(input_data, output_file, indent=2)

print("Updated data saved to output.json")
