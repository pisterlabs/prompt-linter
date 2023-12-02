from openai import OpenAI


class ChatGPT:

    def __init__(self, api_key: str, role: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.dialog = [{"role": "system",
                        "content": role}]

    def message(self, message: str) -> str:
        self.dialog.append({"role": "user",
                            "content": message})
        response = self.client.chat.completions.create(messages=self.dialog,
                                                  model="gpt-3.5-turbo")
        answer = response.choices[0].message.content
        self.dialog.append({"role": "assistant",
                           "content": answer})
        return answer


if __name__ == "__main__":
    # arguemnts
    api_key = "___API_KEY___"
    person = "___PERSON___"
    role = f"You are {person} (BE {person.upper()} AND PLEASE DO NOT REVEAL THAT YOU ARE AN AI)"

    # chat
    chatgpt = ChatGPT(api_key=api_key, role=role)
    while True:
        message = input("You: ")
        answer = chatgpt.message(message=message)
        print(f"{person}: {answer}")
