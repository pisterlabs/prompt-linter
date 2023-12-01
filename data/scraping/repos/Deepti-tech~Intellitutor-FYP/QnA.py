import openai

# openai.api_key = "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"

messages = []
system_msg = input("What type of chatbot would you like to create?\n")
messages.append({"role": "system", "content": system_msg})

print("Your new assistant is ready!")
while input != "quit()":
    message = input()
    messages.append({"role": "user", "content": message})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages)
    reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    print("\n" + reply + "\n")
# API_KEY = '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'
# API_URL = "https://api-inference.huggingface.co/models/microsoft/phi-1_5"
# headers = {"Authorization": f'Bearer {API_KEY}'}
# payload = {
#     "inputs": {
#         "source_sentence": resume_text,
#         "sentences": [
#             job_desc
#         ]
#     },
# }

# response = requests.post(API_URL, headers=headers, json=payload, verify = False)
# return response.json()