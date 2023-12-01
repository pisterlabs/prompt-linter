import openai

with open('C:/Users/AZEEM/Desktop/API.txt') as file:
    api_key = file.read().strip()

openai.api_key = api_key


response=openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a financial advisor."},
        {"role": "user", "content": "What are some tips for saving money?"},
        {"role": "assistant", "content": "Creating a budget, reducing expenses, and saving on utilities are some ways to save money."},
        {"role": "user", "content": "How do I create a budget?"}
    ]
)


print(response['choices'][0]['message']['content'].strip())