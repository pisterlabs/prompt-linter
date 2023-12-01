# import requests
# import json
import os
import textwrap
import openai
import pytz
import datetime
import discord
from io import BytesIO
from tiktoken import encoding_for_model
from parsedatetime import Calendar
from discord.commands import Option
from discord.utils import basic_autocomplete
from constants import *
from gtts import gTTS

calendar = Calendar()

TOKENIZER = encoding_for_model("gpt-3.5-turbo")
get_tokens = lambda text: len(TOKENIZER.encode(text))

class Summary:
    def __init__(self, message):
        self.messages = lambda prompt: [
            {"role": "system", "content": message},
            {"role": "user", "content": prompt},
        ]
        self.full_summary = ""

    def summarize(self, prompt, key):
        if key == "pok its confusing because i dont have diZ context":
            key = os.getenv("CHATGPT_TOKEN")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages(prompt),
            api_key=key,
        )

        assert isinstance(response.choices[0].message.content, str), "API response is not a string"
        summary = response.choices[0].message.content
        self.full_summary += summary + "\n"
        return summary

    def tts(self, username):
        tts = gTTS(text=self.full_summary, lang="en", slow=False)
        sound = BytesIO()
        tts.write_to_fp(sound)
        sound.seek(0)
        return discord.File(sound, filename=f"{username}'s Summary.mp3")


async def summary(
    ctx,
    messages: Option(int, "Number of messages to summarize", required=False, default=100, min_value=1),
    mode: Option(str, "Method of constructing summary, check your modes with /listmodes", required=False, default="standard"),
    channel: Option(discord.TextChannel, "The channel which will be summarized", required=False),
):
    await ctx.response.defer()

    if channel is None:
        channel = ctx.channel

    history = []
    first = True

    async for message in channel.history(limit=None):
        if first:
            first = False
            continue

        history.append(message)

        if len(history) == messages:
            break

    if not history:
        await ctx.followup.send(LESS_MESSAGES)
        return

    await send_summary(ctx, history[::-1], mode, channel)


async def fromtosummary(
    ctx,
    from_time: Option(str, "Start time of messages to summarize, in natural language", required=True),
    to_time: Option(str, "End time of messages to summarize, in natural language", required=True),
    mode: Option(str, "Method of constructing summary, check your modes with /listmodes", required=False, default="standard"),
    channel: Option(discord.TextChannel, "The channel which will be summarized", required=False),
):
    await ctx.response.defer()

    user, _ = setup_user(str(ctx.author), ctx.guild.name, ctx.guild.id)

    if channel is None:
        channel = ctx.channel

    offset = TIMEZONES[user["region"]]
    if offset == 0:
        await ctx.followup.send(f"**NOTE: Currently your timezone is UTC, this may cause some unwanted times, use /setregion if you want to change your timezone.**\n")

    from_time = calendar.parse(from_time)[0]
    to_time = calendar.parse(to_time)[0]
    from_time = datetime.datetime(
        from_time.tm_year,
        from_time.tm_mon,
        from_time.tm_mday,
        from_time.tm_hour,
        from_time.tm_min,
        from_time.tm_sec,
        tzinfo=pytz.FixedOffset(offset * 60),
    )
    to_time = datetime.datetime(
        to_time.tm_year,
        to_time.tm_mon,
        to_time.tm_mday,
        to_time.tm_hour,
        to_time.tm_min,
        to_time.tm_sec,
        tzinfo=pytz.FixedOffset(offset * 60),
    )
    await ctx.send(f"Parsed times: {from_time.strftime('%x %X')} to {to_time.strftime('%x %X')}")
    utc_from_time = from_time.astimezone(pytz.UTC)
    utc_to_time = to_time.astimezone(pytz.UTC)

    history = []
    first = True
    async for message in channel.history(limit=None):
        if first:
            first = False
            continue
        if message.created_at < utc_to_time and message.created_at > utc_from_time:
            history.append(message)
        if message.created_at < utc_from_time:
            break

    if not history:
        await ctx.followup.send(f"There were no messages in the time range, {from_time.strftime('%x %X')} to {to_time.strftime('%x %X')}. Please double check your timezone with /setregion")
        return
    
    await send_summary(ctx, history[::-1], mode, channel)


async def unreadsummary(
    ctx,
    mode: Option(str, "Method of constructing summary, check your modes with /listmodes", required=False, default="standard"),
):
    await ctx.response.defer()

    history = []
    first = True

    async for message in ctx.channel.history(limit=None):
        if first:
            first = False
            continue

        history.append(message)

        if message.author.id == ctx.author.id:
            break

    if not history:
        await ctx.followup.send(LESS_MESSAGES)
        return

    await send_summary(ctx, history[::-1], mode, ctx.channel)


async def send_summary(ctx, messages, mode, channel=None):
    if len(messages) > 5000:
        await ctx.followup.send("The maximum number of messages that can be summarized is 5000.")
        return
    # Grab variables from database
    user, server = setup_user(str(ctx.author), ctx.guild.name, ctx.guild.id)
    language = user["language"]
    if mode in user["modes"]:
        mode_prompt = user["modes"][mode]
    else:
        await ctx.followup.send(f"Mode `{mode}` not found. Please try again with a valid mode. Use /listmodes to see all your modes.")
        return
    thread = user["thread"]
    api_key = user["api-key"]
    if api_key == "NONE":
        api_key = server["api-key"]
        if api_key == "NONE":
            await ctx.followup.send(f"Due to inappropriate usage of the common API key provided by Summary Bot, we have decided to shut it down. Summary Bot is still fully functional, but just with your own OpenAI API key.")
            return

    if mode not in user["modes"]:
        await ctx.followup.send(f"Mode `{mode}` not found. Please try again with a valid mode. Use /listmodes to see all your modes.")
        return

    # Group messages into groups of MAX_TOKENS
    # Create a list starts for the first message in each group
    next_group = True
    groups = []
    starts = []
    curr_group = ""
    in_token_count = 0

    for message in messages:
        if next_group:
            starts.append(message)
            next_group = False

        m = f"{message.author.display_name}: {message.content}\n"

        if get_tokens(curr_group + m) > MAX_TOKENS:
            in_token_count += get_tokens(curr_group)
            groups.append(curr_group)
            curr_group = ""
            next_group = True

        curr_group += m

    starts.append(message)
    groups.append(curr_group)
    in_token_count += get_tokens(curr_group)

    if api_key == "NONE":
        user["in_token_count"] += in_token_count
        set_user(str(ctx.author), user)

        server["in_token_count"] += in_token_count
        set_server(ctx.guild.name, ctx.guild.id, server)

    headings = []
    for i in range(len(starts) - 1):
        message1_link = MESSAGE_LINK.format(ctx.guild.id, channel.id, starts[i].id)
        message2_link = MESSAGE_LINK.format(ctx.guild.id, channel.id, starts[i + 1].id)
        num_messages = groups[i].count("\n")
        headings.append(f"Summary from **{message1_link}** to **{message2_link}**. This summary contains __{num_messages}__ messages.\n\n")

    print(f"{ctx.author}'s summary: Grouped into {len(groups)} groups")
    print(f"{ctx.author}'s summary of {len(messages)} messages")

    # Start sending the summary
    info_str = "\n"
    info_str += f"Messages: ``{len(messages)}``\n"
    info_str += f"Mode: ``{mode}``\n"
    info_str += f"Language: ``{language}``\n"
    info_str += f"Thread: ``{thread}``\n"
    info_str += f"Total Input Tokens: ``{in_token_count}``\n"
    info_str += f"By {ctx.author.mention}"
    embed = discord.Embed(title="Generated summary")
    embed.description = f"A `{len(groups)}-message` summary is being prepared..."
    embed.add_field(name="Command Arguments:", value=info_str)
    summary = Summary(INTRO_MESSAGE.format(ctx.guild.name, mode_prompt, language))

    try:
        if thread:
            message = await ctx.followup.send(embed=embed)
            thread = await ctx.channel.create_thread(name=f"Summary by {ctx.author.display_name}, {len(messages)} messages", message=message)
        else:
            message = await ctx.followup.send(embed=embed)
            thread = channel

        for i in range(len(groups)):
            response = summary.summarize(groups[i], api_key)
            # Send message in 2000 character chunks to prevent error
            block = headings[i] + response
            block = block.replace("\n", "[{%$%*0*@)#*$J)SDJF*J#}]")
            for chunk in textwrap.wrap(headings[i] + response, 2000):
                chunk = chunk.replace("[{%$%*0*@)#*$J)SDJF*J#}]", "\n")
                await thread.send(chunk, suppress=True)

            # Update token count
            if api_key == "NONE":
                tokens = get_tokens(response)
                user["out_token_count"] += tokens
                set_user(str(ctx.author), user)

                server["out_token_count"] += tokens
                set_server(ctx.guild.name, ctx.guild.id, server)

            print(f"{ctx.author}'s summary: Sent group {i+1}/{len(groups)}")

        # Send TTS summary
        try:
            result = summary.tts(ctx.author.display_name)
            await thread.send(file=result)
        except Exception as e:
            print(e)
            await thread.send("There was an error in generating the TTS summary")

    except Exception as e:
        print(e)
        await message.edit(content=ERROR.format(e))
