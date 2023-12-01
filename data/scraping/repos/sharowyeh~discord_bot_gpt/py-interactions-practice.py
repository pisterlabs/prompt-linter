#NOTE: just for discord api & package practice,
#      i think pycord is more suitable for my usage
import os
import interactions
import openai

if os.environ.get('DISCORD_BOT_TOKEN') is None:
    print('bot token can not be empty')
    exit()

if os.environ.get('OPENAI_API_KEY') is None:
    print('openai api key can not be empty')
    exit()

openai.api_key = os.environ.get('OPENAI_API_KEY')

# given bot token from developer site bot page(can only renew for display)
# can leave unset guild id to the default_scope,
# that discord client app no need to enable developer mode as well...
# let bot interaction where invited in(perhaps? probably?)
bot = interactions.Client(
    token=os.environ.get('DISCORD_BOT_TOKEN'),
)

if os.environ.get('DEFAULT_GUILD_ID') is None:
    print('use token default scope')
else:
    # I need method to change client params
    bot = interactions.Client(
        token=os.environ.get('DISCORD_BOT_TOKEN'),
        default_scope=os.environ.get('DEFAULT_GUILD_ID'),
    )

#@bot.command(
#    name="say_hello",
#    description="let bot say hello world for debug",
#)
# can simplify as below
@bot.command()
async def say_hello(ctx: interactions.CommandContext):
    """let bot say hello world for debug"""
    await ctx.send("hello world")

@bot.command()
@interactions.option()
async def echo_me(ctx: interactions.CommandContext, text: str):
    """let bot echo your input"""
    await ctx.send(f"reply: '{text}' bot latency: '{round(bot.latency)} ms'")

@bot.command()
@interactions.option()
async def chat_me(ctx: interactions.CommandContext, text: str):
    """talk something with open ai"""
    # reply message first for context prevent task terminated before openai response(I guess?)
    await ctx.send(f"said '{text}'")

    # let openai API call by async? but completion does not have acreate method
    # or use from asgiref.sync import sync_to_async? [link](https://github.com/openai/openai-python/issues/98)
    print(f"q: {text}")
    token_length = 100
    resp = openai.Completion.create(
        engine="text-davinci-003",
        prompt=text,
        temperature=0.5,
        max_tokens=token_length,
#        frequency_penalty=0,
#        presence_penalty=0
#        stream=False,
#        stop="\n",
    )
    reply_text = ""
    while True:
        print(resp)
        if not hasattr(resp, 'choices') or len(resp.choices) == 0:
            await ctx.send("I got no response")
            break
        if not resp.choices[0].text:
            await ctx.send("I got empty response")
            break
        reply_text = resp.choices[0].text
        print(f"a: {reply_text}")
        await ctx.send(f"{reply_text}")
        
        if not resp.choices[0].finish_reason or resp.choices[0].finish_reason != "length":
            print(f"Response may end")
            break;

        # append text for rest of responses(is necessary?)
        text += reply_text
        # increase token length
        token_length += 100
        resp = openai.Completion.create(
            engine="text-davinci-003",
            prompt=text,
            temperature=0.5,
            max_tokens=token_length,
#            stream=False,
#            stop="\n",
        )

    await ctx.send(f"hope I answered...")
    print("==== end of resp ====")

bot.start()
