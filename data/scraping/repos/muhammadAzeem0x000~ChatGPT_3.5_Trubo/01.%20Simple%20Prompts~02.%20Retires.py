import openai

with open('C:/Users/AZEEM/Desktop/API.txt') as file:
    api_key = file.read().strip()

openai.api_key = api_key

response=openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's a good book to read on a rainy day?"}
    ],
  n=3
)

print(response)


print("Now Starting the For Loop ---------------------------")
for i in (response["choices"]):
  print(i["message"]['content'].strip()) 
  print(" ////////  ")