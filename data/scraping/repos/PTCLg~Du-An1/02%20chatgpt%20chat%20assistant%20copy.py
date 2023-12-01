import openai

openai.api_key = "bggoE2FdeHrX-GP0laq1s5Z8SaZ1iR-Q9ZEuWqOQDhsajy3uGccB1-Mi5xDn48yAHXSeCw."

messages = []
system_msg = input("Bạn muốn tạo loại chatbot nào?\n")
messages.append({"role": "system", "content": system_msg})

print("Lý Hành của bạn đã sẵn sàng!")
while True:
    message = input()
    if message == "quit()":
        break
    messages.append({"role": "user", "content": message})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages)
    reply = response.choices[0].message['content']
    messages.append({"role": "assistant", "content": reply})
    print("\n" + reply + "\n")
