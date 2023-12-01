import openai

f = open("/home/arjun/Desktop/Some docuements/OpenAiKey.txt",'r')
key = f.read()[:-1]
openai.api_key = key
f.close()
while True:
    prompt = input("Enter prompt: ")
    response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0.7, max_tokens=30)
    print(response["choices"][0]["text"])
    # print(response["choices"][0])
