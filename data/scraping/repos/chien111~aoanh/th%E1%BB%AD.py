import openai
openai.api_key = "YOUR_API_KEY"
prompt = "Hello, how can I help you today?"
response = openai.Completion.create(engine="text-davinci-002",prompt=prompt,max_tokens=60)

print(response.choices[0].text)
"python.envFile": "${workspaceFolder}/.env",
