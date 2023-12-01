import asyncio
import json
import random
import re
from datetime import datetime, timedelta

import discord
import pytz
import requests
from discord import Intents
from discord import Embed
from discord.ext import commands, tasks
from flashtext import KeywordProcessor

import nerdprice
import nerdnews
from database import *
from nerdchat import get_completion, purge_conversation_history
from nerdconfig import *
from nerdmail import *

intents = discord.Intents.all()
bot = commands.Bot(command_prefix='!', intents=intents)

COOLDOWN_TIME = timedelta(seconds=5)  # Adjust as needed
user_cooldowns = {}


def check_cooldown(user_id):
    current_time = datetime.now()

    if user_id in user_cooldowns:
        last_message_time = user_cooldowns[user_id]
        if current_time - last_message_time < COOLDOWN_TIME:
            return True
        else:
            user_cooldowns[user_id] = current_time
            return False
    else:
        user_cooldowns[user_id] = current_time
        return False


@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    guild = bot.get_guild(SERVER_ID)
    print(f'Server ID {guild} {SERVER_ID}')
    await update_status()  # Call the function directly to update status immediately
    update_status.start()
    check_arch_news.start()


@bot.event
async def on_message(message):
    # Process commands first
    await bot.process_commands(message)

    reaction_words = ["fuck", "ontario", "ontari-o", "methswick", "toronto"]
    reactions = ["<:goat:1072602820206931988>", "<:suschicken:905215028226043914>", "<:kappa:873946053630636052>"]
    
    for word in reaction_words:
        if word in message.content.lower() and random.random() < 0.25:
            try:
                random_reaction = random.choice(reactions)
                await asyncio.sleep(1)
                await message.add_reaction(random_reaction)
                break
            except Exception as e:
                print(e)
                pass

    if bot.user.mentioned_in(message):
        message_content = message.content.replace(
            f'<@{bot.user.id}>', '').replace(f'<@!{bot.user.id}>', '').strip()
        if not check_cooldown(message.author.id):
            # If the bot is tagged or replied to, get a completion from OpenAI and send it as a response
            guild = bot.get_guild(SERVER_ID)
            member = guild.get_member(message.author.id)

            # User got past the cooldown check, make sure they've been a member long enough
            if datetime.now(pytz.UTC) - member.joined_at > ONE_MONTH:
                await asyncio.sleep(1)
                async with message.channel.typing():  # This shows the bot as typing
                    try:
                        response = get_completion(
                            message_content, message.author.id)
                    except Exception as e:
                        response = "Something went wrong, sorry."
                        print(e)
                    await message.channel.send(response)
        else:
            await message.channel.send("You must wait 5 seconds between prompts.")

    # If the message is positively not a DM
    if not isinstance(message.channel, discord.DMChannel):

        # Load the filters
        conn = create_connection()
        filters = load_filters(conn)

        # Check each filter
        for user_id, filter_string, channel_id in filters:
            keyword_processor = KeywordProcessor(case_sensitive=False)
            keyword_processor.add_keyword(filter_string)

            found_keywords = keyword_processor.extract_keywords(
                message.content)

            # If the filter string is found in the message content and the message is in the correct channel
            if found_keywords and (channel_id == '0' or str(message.channel.id) == channel_id):
                # Get the User object
                user = bot.get_user(int(user_id))

                # If the User object exists
                if user:
                    # Send a DM
                    message_url = f"https://discord.com/channels/{message.guild.id}/{message.channel.id}/{message.id}"
                    await user.send(f"A message containing \"{filter_string}\" was just posted by {message.author} in <#{message.channel.id}>. You can view it here: {message_url}\n\nMessage Content:\n{message.content}")

        close_connection(conn)


@bot.command(
    name='nbnerds',
    help='Creates an email account for the specified username.',
    brief='Creates an email account',
    aliases=['create_email']
)
async def nbnerds(ctx, username):
    if not isinstance(ctx.channel, discord.DMChannel):
        await ctx.send("Please send me a DM, write !nbnerds uSeRnAmE and I will create your email.")
        return

    guild = bot.get_guild(SERVER_ID)
    member = guild.get_member(ctx.author.id)

    if member is None or datetime.now(pytz.UTC) - member.joined_at < ONE_MONTH:
        await ctx.send('You must be a member of the server for over one month to use this command.')
        return

    emails = load_emails()
    if str(ctx.author.id) in emails:
        await ctx.send('You have already created an email.')
        return

    if not re.fullmatch(USERNAME_REGEX, username) or len(username) >= 30 or len(username) <= 2:
        await ctx.send('Invalid username. It must only contain alphanumeric characters and be shorter than 31 characters and longer than 2 characters.')
        return

    # Check if the username is reserved
    if username.lower() in RESERVED_USERNAMES:
        await ctx.send('The username you have chosen is reserved. Please choose a different username.')
        return

    print("User invoked email creation command.")

    # Get a random password
    response = requests.get(
        f'{MAIL_API_URL}accounts/random_password/', headers=HEADERS)

    if response.status_code != 200:
        await ctx.send('There was an issue generating the password. Please try again later.')
        return

    password = response.json().get('password')

    # Prepare the data for our post request
    data = {
        "username": f"{username}@nbnerds.ca",
        "password": password,
        "role": "SimpleUsers"
    }

    # Convert python dictionary to json
    data = json.dumps(data)

    # Make the post request
    response = requests.post(
        f'{MAIL_API_URL}accounts/', headers=HEADERS, data=data)

    # Check if the request was successful
    if response.status_code == 201:
        account_pk = response.json()['pk']
        emails[str(ctx.author.id)] = f"{username}@nbnerds.ca"
        save_emails(emails)
        await ctx.send(f'Email account {username}@nbnerds.ca successfully created! The password is {password}. You may change your password and other settings at the webmail server.')
        await ctx.invoke(bot.get_command('mail'))
    else:
        await ctx.send('There was an issue creating the email account. Please try again later.')


@bot.command(
    name='resetpw',
    help='Resets the password of your email account.',
    brief='Resets email account password',
    aliases=['password_reset']
)
async def resetpw(ctx):
    if not isinstance(ctx.channel, discord.DMChannel):
        await ctx.send("Please send me a DM and write !resetpw. I will reset your password.")
        return

    emails = load_emails()
    user_email = emails.get(str(ctx.author.id))

    if user_email is None:
        await ctx.send('You have not created an email account.')
        return

    # Retrieve account pk
    response = requests.get(
        f'{MAIL_API_URL}accounts/?search={user_email}', headers=HEADERS)

    if response.status_code != 200 or not response.json():
        await ctx.send('There was an issue retrieving the account details. Please try again later.')
        return

    account_pk = response.json()[0].get('pk')

    # Get a random password
    response = requests.get(
        f'{MAIL_API_URL}accounts/random_password/', headers=HEADERS)

    if response.status_code != 200:
        await ctx.send('There was an issue generating the password. Please try again later.')
        return

    new_password = response.json().get('password')

    # Prepare the data for our patch request
    data = {
        "password": new_password,
    }

    # Convert python dictionary to json
    data = json.dumps(data)

    # Make the patch request
    response = requests.patch(
        f'{MAIL_API_URL}accounts/{account_pk}/', headers=HEADERS, data=data)

    # Check if the request was successful
    if response.status_code == 200:  # HTTP status code for 'OK', as per REST conventions
        await ctx.send(f'Password for {user_email} successfully reset! The new password is {new_password}. You may change your password and other settings at the webmail server.')
    else:
        await ctx.send(f'There was an issue resetting the password. Please try again later. {response} {new_password} {account_pk} {data} {headers}')


@bot.command(
    name='mail',
    help='Provides instructions on how to connect to your email account.',
    brief='Provides email account instructions',
    aliases=['instructions']
)
async def mail(ctx):
    await ctx.send(MAIL_HELP_MESSAGE)


@bot.command()
async def nbnerdshelp(ctx):
    await ctx.send(HELP_HELP_MESSAGE)

@bot.command(
    name='purge',
    help='Purges conversation history.'
)
async def purge(ctx):
    purge_conversation_history(ctx.author.id)
    await ctx.send('Your conversation history is reset.')


@bot.command()
async def delete_msg(ctx, channel_id: int, msg_id: int):
    # (user: dnim)
    user_id = 601774566359957504

    if ctx.author.id != user_id:
        await ctx.send('You do not have permission to use this command.')
        return

    channel = bot.get_channel(channel_id)
    if not channel:
        await ctx.send('Channel not found.')
        return

    try:
        msg = await channel.fetch_message(msg_id)
    except discord.NotFound:
        await ctx.send('Message not found.')
        return

    await msg.delete()
    await ctx.send(f"Message with ID {msg_id} has been deleted.")


@bot.command()
async def say(ctx, channel_id: int, *, message):
    # (user: dnim)
    user_id = 601774566359957504

    if ctx.author.id != user_id:
        await ctx.send('You do not have permission to use this command.')
        return

    channel = bot.get_channel(channel_id)
    if not channel:
        await ctx.send('Channel not found.')
        return

    await channel.send(message)


@bot.command(
    name='filter',
    help="Manages your message filters, use in DMs.\n\n"
         "Usage:\n"
         "!filter create <string> <channel_id> - Creates a new filter.\n"
         "!filter delete <string> <channel_id> - Deletes an existing filter.\n"
         "!filter view - Lists all your filters.\n\n"
         "<channel_id> is the ID of the channel the filter is for (or '0' for the entire server).\n"
         "<string> is the text you want to filter for."
)
async def filter(ctx, action, *args):
    if not isinstance(ctx.channel, discord.DMChannel) and action.lower() != 'help':
        await ctx.send("Please configure your filters in a DM with me.")
        return

    conn = create_connection()
    initialize_db(conn)

    user_id = str(ctx.author.id)

    if action.lower() == 'create':
        if args:
            # The last argument is treated as the channel_id, and the rest as the string
            channel_id = args[-1]
            # Join all other arguments with a space
            string = " ".join(args[:-1])

            # Check if the channel ID exists
            if channel_id != '0':
                channel = bot.get_channel(int(channel_id))
                if not channel:
                    await ctx.send("The channel ID provided does not exist.")
                    return

            if len(string) < 3:
                await ctx.send('Your filter string must be at least 3 characters long.')
                return
            elif len(string) > 256:
                await ctx.send('Your filter string must be no longer than 256 characters.')
                return

            # Check for special characters
            if not string.isalnum() and not ' ' in string:
                await ctx.send("Your filter string should contain only alphanumeric characters and spaces.")
                return

            filters = load_filters(conn)
            for filter in filters:
                existing_user_id, existing_string, existing_channel_id = filter
                if existing_string == string and existing_user_id == user_id:
                    if existing_channel_id == '0' and channel_id != '0':
                        await ctx.send('A filter with the same string already exists for the entire server.')
                        return
                    elif existing_channel_id != '0' and channel_id == '0':
                        await ctx.send(f'A filter with the same string already exists for channel <#{existing_channel_id}>.')
                        return

            save_filters(conn, user_id, string, channel_id)
            await ctx.send('Filter created successfully!')

        else:
            await ctx.send('Please provide a valid string and channel ID.')

    elif action.lower() == 'delete':
        if args:
            # The last argument is treated as the channel_id, and the rest as the string
            channel_id = args[-1]
            # Join all other arguments with a space
            string = " ".join(args[:-1])
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM filters WHERE user_id=? AND filter_string=? AND channel_id=?
            ''', (user_id, string, channel_id))
            conn.commit()
            if cursor.rowcount > 0:
                await ctx.send('Filter deleted successfully!')
            else:
                await ctx.send('No matching filter found.')
        else:
            await ctx.send('Please provide a valid string and channel ID.')

    elif action.lower() == 'view':
        cursor = conn.cursor()
        cursor.execute(
            "SELECT filter_string, channel_id FROM filters WHERE user_id=?", (user_id,))
        filters = cursor.fetchall()

        if filters:
            response = 'Here are your filters:\nCopy paste the filter string into the `!filter delete` command to delete it.\n\n'
            for filter_string, channel_id in filters:
                response += f'{filter_string} {channel_id}\n'
            await ctx.send(response)
        else:
            await ctx.send('You have no filters set up.')

    elif action == "help" or action is None:
        await ctx.send(FILTER_HELP_MESSAGE)
        return

    close_connection(conn)

from discord import Embed

@bot.command(name='divvy', help='Shows the ex-dividend and payout dates for LBS')
@commands.cooldown(1, 3600, commands.BucketType.default)
async def divvy(ctx):
    # Check if the command is being called from the specified channel
    if ctx.channel.id != BOTSPAM:
        return

    dividend_dates = nerdprice.get_dividend_dates()

    if dividend_dates:
        # Create a new embed with a title and a color
        embed = Embed(title="LBS Divvies", description="Details of the upcoming dividend dates for LBS.", color=0x2ECC71)

        sorted_dates = sorted(dividend_dates, key=lambda x: x[0])
        current_date = datetime.now().date()
        next_ex_dividend = sorted_dates[0][0]
        next_payout = sorted_dates[0][1]
        ex_dividend_countdown = (next_ex_dividend - current_date).days
        payout_countdown = (next_payout - current_date).days

        # Bold the header for the next dividends and italicize the countdown
        embed.add_field(name="**Next Ex-Divvy Date**", value=f"{next_ex_dividend} - *in {ex_dividend_countdown} days*", inline=True)
        embed.add_field(name="**Next Payout Date**", value=f"{next_payout} - *in {payout_countdown} days*", inline=True)
        
        # Add a separator for clarity
        embed.add_field(name="\u200b", value="\u200b", inline=True)  

        # Add other future dates in numeric format
        further_dates = "\n".join([f"Ex-Dividend: {date[0].strftime('%Y-%m-%d')}, Payout: {date[1].strftime('%Y-%m-%d')}" for date in sorted_dates[1:]])
        if further_dates:
            embed.add_field(name="Future Divvies", value=further_dates, inline=False)

        await ctx.send(embed=embed)
    else:
        await ctx.send("There was an issue fetching the dividend dates. Please try again later.")


@bot.command(name='archnews', help='Follow or unfollow Arch Linux news')
async def archnews(ctx, action):
    # Check if the command is being called from the specified channel or inside a DM
    if ctx.channel.id != BOTSPAM and not isinstance(ctx.channel, discord.DMChannel):
        return
    if action.lower() == 'follow':
        nerdnews.add_user(ctx.author.id)
        await ctx.send('You will now be notified of new Arch Linux news.')
    elif action.lower() == 'unfollow':
        nerdnews.remove_user(ctx.author.id)
        await ctx.send('You will no longer be notified of new Arch Linux news.')
    else:
        await ctx.send('Please specify either "follow" or "unfollow".')


@tasks.loop(minutes=10)  # check every 10 minutes
async def check_arch_news():
    unprocessed_news = nerdnews.get_unprocessed_news()
    users_to_notify = nerdnews.get_users()

    if unprocessed_news:
        mentions = ' '.join([f'<@{user_id}>' for user_id in users_to_notify])
        channel = bot.get_channel(TECHTALK)
        
        if not channel:
            print(f"Channel with ID {channel_id} not found.")
            return
        
        for title, link in unprocessed_news:
            embed = Embed(
                title="New Arch News",
                description=f"[{title}]({link})",
                color=0x7289DA,  # Discord's blurple
                timestamp=datetime.utcnow()  # current time as the embed timestamp
            )

            await channel.send(content=f'{mentions}', embed=embed)


@tasks.loop(seconds=10)
async def update_status():
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.watching,
        name=nerdprice.get_next_status()))

@update_status.before_loop
async def before_update_status():
    # Pause the loop on first run
    await asyncio.sleep(10)


bot.run(DISCORD_BOT_TOKEN)
