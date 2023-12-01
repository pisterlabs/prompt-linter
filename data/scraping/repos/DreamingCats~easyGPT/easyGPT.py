import openai
openai.api_key = ""  #fill in your api_key

question=input("Input your question:")
print("chatGPT says:")

answer = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    max_tokens=1024,
    temperature=0.7, #越接近1重复率越低
    messages=[
        {"role": "user", "content": question}
    ],
    
)

print(answer['choices'][0]['message']['content'])
input('\nPress Enter to exit...')
