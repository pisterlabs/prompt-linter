import openai

openai.api_key = "KEY"

messages = []
system_msg = input("What type of chatbot would you like to create?\n")
messages.append({"role": "system", "content": system_msg})

print("Your new assistant is ready!")
while True:
    user_input = input("You: ")  # Get user input
    if user_input.lower() == "quit()":
        break

    messages.append({"role": "user", "content": user_input})
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages)
    
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    print("Assistant: " + reply + "\n")