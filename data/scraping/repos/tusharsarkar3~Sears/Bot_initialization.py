from openai import OpenAI
from Intial_contexts import *

OPENAI_KEY = 'API KEY'

client = OpenAI(api_key=OPENAI_KEY)


class OpenAIBot:
    def __init__(self, bot_name):
        self.bot_name = bot_name
        self.conversation_history = []

    def ask(self, message):
        self.conversation_history.append({"role": "user", "content": message + 'You are going to output in JSON'})
        response = client.chat.completions.create(
            response_format={"type": "json_object"},
            model="gpt-3.5-turbo-1106",
            messages=self.conversation_history
        )
        bot_response = response.choices[0].message.content

        # print('Bot Response----->>>>',type(bot_response))
        # print('Bot response --->', bot_response['content'])
        #.message['content']
        self.conversation_history.append({"role": "assistant", "content": bot_response})
        return bot_response
    
        # completion = client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #         {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
        #         {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
        #     ]
        # )
def simulate_conversation(c1, bot1, bot2, start_message):
    c1.conversation_history.append(start_message)
    print(f"{bot1.bot_name}: {start_message}")
    reply = bot1.ask(start_message)
    # print(reply)
    print(f"{bot2.bot_name}: {reply}")
    for _ in range(5):  # Number of exchanges
        # c_reply = c1.ask(reply)
        # print("Context bot question:")
        # print(c_reply)
        reply = bot2.ask(reply+"Ask cross question")
        print(f"{bot1.bot_name}: {reply}")
        reply = bot1.ask(reply)
        print(f"{bot2.bot_name}: {reply}")


# Initialize bots

context_bot = OpenAIBot("ContextBot")
context_bot.ask(about_business) #providing context regarding the current business situation

pricing_bot = OpenAIBot("PricingBot")
pricing_bot.ask(pricing_context)

analytics_bot = OpenAIBot("AnalyticsBot")
analytics_bot.ask(analytics_context)

# Start the simulated conversation
simulate_conversation(context_bot, pricing_bot, analytics_bot, "How should we adjust our pricing strategy today?")
