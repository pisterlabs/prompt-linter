from openai_afc import AutoFnChatCompletion, AutoFnDefinition, AutoFnParam

def add(x, y):
    print(f'Adding {x} and {y}')
    return x + y

completion = AutoFnChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "What is 42 + 99?"}
        ],
        functions=[
            AutoFnDefinition(
                add,
                description='Add two numbers',
                params=[
                    AutoFnParam('x', {'type': 'number'}),
                    AutoFnParam('y', {'type': 'number'})
                ]
            )
        ],
        temperature=0.0
    )
# This completion response is of the same form you would get from openai.ChatCompletion
print(completion['choices'][0]['message']['content'])