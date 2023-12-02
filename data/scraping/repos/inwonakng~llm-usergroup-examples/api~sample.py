import os
os.environ['OPENAI_API_KEY']="sk-111111111111111111111111111111111111111111111111"
os.environ['OPENAI_API_BASE']="http://0.0.0.0:5001/v1"
import openai

prompt = [
    {
        'role': 'user',
        'content': 'You are a helpful assistant. You will answer questions I ask you. Reply with Yes if you understand.'
    },{
        'role': 'assistant',
        'content': 'Yes, I understand'
    },{
        'role': 'user',
        'content': 'What color is the sky?'
    }
]
response = openai.ChatCompletion.create(
    model="x",
    messages = prompt
)
output = response['choices'][0]['message']['content']
print('Model output:', output)
