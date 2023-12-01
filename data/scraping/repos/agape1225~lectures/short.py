import openai

openai.api_key = "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" 

messages = []
    
while True :
    prompt = input()
    if prompt == "." : break
    print("Me : " + prompt)    

    messages.append({"role": "user", "content": prompt})
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  messages=messages)

    res= completion.choices[0].message['content']
    messages.append({"role": 'assistant', "content": res}  )

    print(messages)

    print("GPT : " +res)

    