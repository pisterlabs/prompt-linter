import openai


class Person:
    def __init__(self):

        self.chat_memory = []
        self.persona = str()
        #self.model = "gpt-3.5-turbo-16k-0613"
        self.model = "gpt-3.5-turbo-16k-0613"

    def give_context(self, context: str):

        if len(context) <= 0:
            raise Exception(
                f"FATAL: initializing memory of {self.__class__.__name__} with no context.")

        ctx = {
            "role": "user",
            "content": context
        }

        self.chat_memory.append(ctx)

    def reply(self, msg: str):

        # test: frequently remember ai of what she is supposed to do
        self.chat_memory.append(self.chat_memory[0])

        ctx = {
            "role": "user",
            "content": msg
        }

        self.chat_memory.append(ctx)
        return self.say()




    def say(self):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.chat_memory,
            temperature=0,
            max_tokens=64
        )

        ctx = {
            "role": response.choices[0].message.role,
            #"content": f"{self.persona}: {response.choices[0].message.content}"
            "content": f"{response.choices[0].message.content}"
        }

        self.chat_memory.append(ctx)
        return  response.choices[0].message.content
