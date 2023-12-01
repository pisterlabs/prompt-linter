import openai

class Toggler:
    def __init__(self, thinker):
        self.description = """
Use this class to switch which model you are using
arguments: thinker (Thinker)
        """
        self.thinker = thinker
        self.prompt = [
            {
                'role':'user',
                'content':"You are a monitor process for a virtual assistant. You should respond to every input with 'N/A' and nothing else, unless the user requests that a particular model be used. In this case, you should respond 'TOGGLE' and nothing more. Do not under any circumstances respond with anything other than 'TOGGLE' or 'N/A'."
            },
            {
                'role':'assistant',
                'content': 'N/A',
            },
            {
                'role': 'user',
                'content': 'I want to use GPT-4.'
            },
            {
                'role':'assistant',
                'content':'TOGGLE'
            },
        ]
    def process(self, message):
        response = self.check(message)
        # print(response)
        self.react(response, message)

    def check(self, message):
        _message = {'role': 'user', 'content': message}
        
        if 'toggle' in message.lower():
            return 'TOGGLE'
        else:
            return "N/A"
        #TODO: error handle like any other call (write one function?)
#         response = openai.ChatCompletion.create(
#             model= "gpt-3.5-turbo-0301",
#             messages=self.prompt + [_message],
#             temperature=1,
#             top_p=0,

#         )
#         # print(response)
#         return response['choices'][0]['message']['content']


    def react(self, response, message):
        message = message.lower()
        if 'gpt-3' in message:
            model ='gpt-3.5-turbo-0301'
        if 'gpt-4' in message or 'gpt for' in message:
            model = 'gpt-4-0314'
        else:
            model = 'gpt-3.5-turbo-0301'

        if response.strip() == 'TOGGLE':
            print(f'toggling to {model}')
            self.thinker.model = model
        pass