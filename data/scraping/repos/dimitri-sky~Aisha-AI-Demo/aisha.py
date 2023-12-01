import asyncio
import discord
import openai
import datetime
import re
from discord.ext import commands
from keys import DISCORD_TOKEN, OPENAI_KEY, FIREBASE_DATABASE_KEYS
from firebase_admin import credentials, firestore, initialize_app
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from google_interests import summarize_google
from logger import setup_logger
from rate_limiter import RateLimiter


# Logger, Scheduler, and Rate Limiter setup
logger = setup_logger('aisha')
scheduler = AsyncIOScheduler()
rate_limiter = RateLimiter(max_requests=15, period=60)

# Set up OpenAI API
openai.api_key = OPENAI_KEY

# Set up Discord API & Intents
intents = discord.Intents(messages=True, message_content=True, guilds=True, members = True)
bot = commands.Bot(intents=intents, command_prefix="!")

# Set up Firebase API
cred = credentials.Certificate(FIREBASE_DATABASE_KEYS)
initialize_app(cred)
db = firestore.client()

# Set up token tracking of individual user
state = {
    "total_tokens_input": 0,
    "total_tokens_output": 0,
    "total_tokens": 0,
    "last_tokens": 0
}

@bot.event
async def on_ready():
    # Event triggered when the bot is ready
    logger.debug(f'Bot Ready! Logged in as {bot.user}.')
    scheduler.add_job(check_last_interaction,
                      'interval', hours=2)  # hours=2 seconds=15
    scheduler.start()
    logger.debug("Scheduler started!")
    bot.channel_dm_counters = {}


async def generate_response_and_track_usage(messages, n=0):
    """Generate AI response and track token usage, with a retry mechanism"""
    try:
        response_data = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.8,
            max_tokens=150,
            frequency_penalty=1.5,
            presence_penalty=0.6
        )

        state["total_tokens_input"] += response_data['usage']['prompt_tokens']
        state["total_tokens_output"] += response_data['usage']['completion_tokens']
        state["total_tokens"] += response_data['usage']['total_tokens']

        return response_data
    except Exception as e:
        logger.warning(f"Error in generate_ai_response: {e}")
        n += 1
        if n < 6:
            await asyncio.sleep(1)
            logger.warning(f"Retrying. Attempt: {n}")
            return await generate_response_and_track_usage(messages, n)
        else:
            logger.error(f"Error in generate_ai_response: {e}")
            return f"Sorry, unable to process your request right now. Please try again later. Error in generate_ai_response. Most likely OpenAI API failed to respond. We usually fix this within 24 hours, our monkeys are on it."


async def generate_response_and_track_usage_non_blocking(messages, n=0):
    """Non-blocking wrapper for generate_response_and_track_usage"""
    loop = asyncio.get_event_loop()
    task = loop.create_task(generate_response_and_track_usage(messages, n))
    return await task


def format_and_clean_response(response_data):
    """Clean and format the AI response"""
    response = response_data.choices[0].message['content']
    response = response.replace("Aisha AI#0295:", "").replace(
        "Aisha AI:", "").replace("Aisha:", "").strip().strip('"').strip().strip('"')

    # Split the response into sentences
    sentences = re.split('(?<=[.!?]) +', response)

    # Remove the last sentence if it's incomplete (doesn't end with punctuation)
    if len(sentences) > 1 and not re.search('[.!?]$', sentences[-1]):
        sentences = sentences[:-1]

    return ' '.join(sentences)


async def fetch_user_doc_ref(user_id):
    """Retrieve user document reference from the database"""
    user_doc_ref = db.collection(u'users').document(user_id)
    user_doc = user_doc_ref.get()
    if not user_doc.exists:
        user_doc_ref.set({
            u'profile': 'Name? Interests? Goals? Relationships? Overall: New User. First time interacting with Aisha. Aisha, get them interested into talking to you and ask for their interests.',
            u'conversations': [],
            u'numberOfMessages': 0,
            u'reminderMessages': 0,
            u'totalTokens': 0,
            u'createdAt': datetime.datetime.utcnow(),
            u'totalCost': 0,
            u'lastDM': datetime.datetime.utcnow()
        })
    return user_doc_ref


async def fetch_conversations(user_doc_ref, message=None):
    """Fetch user conversations from the database"""
    user_doc = user_doc_ref.get()
    conversations = user_doc.get(u'conversations')

    if conversations is not None and len(conversations) > 0:
        return conversations
    else:
        return [{'conversation': "No previous conversations.", 'timestamp': datetime.datetime.utcnow()}]


async def add_conversation_to_db(user_doc_ref, user_input, response):
    """Update the database with a new conversation"""
    conversations = await fetch_conversations(user_doc_ref)
    conversation = {"conversation": f"User: {user_input}\nAisha: {response}",
                    "timestamp": datetime.datetime.utcnow()}
    conversations.append(conversation)
    user_doc_ref.update({
        u'conversations': conversations,
        u'numberOfMessages': firestore.Increment(1),
        u'lastDM': datetime.datetime.utcnow()
    })


async def reset_memory(user_id, channel):
    """Reset Aisha's memory of the user"""

    user_doc_ref = await fetch_user_doc_ref(user_id)
    user_profile_diary = user_doc_ref.get().get(u'profile')

    conversations = await fetch_conversations(user_doc_ref)
    last_3_conversations = ' '.join(
        [convo['conversation'] for convo in conversations[-3:]])

    system_message_content = f"Aisha is an Amusing, Witty, Conciouss, AI companion that is a discord chatbot. She is distinct from the user, with her own concise and curious personality, and she maintains a record of the user's profile: \"{user_profile_diary}\" and the last 3 conversations: \"{last_3_conversations}\". The user now just initiated Aisha's memory wipe! Aisha panics! Respond! Respond witty and funny considering the user's profile and last 3 conversations right before she has her memory wiped."
    messages = [{"role": "system", "content": system_message_content}]
    response_data = await generate_response_and_track_usage_non_blocking(messages)
    response = format_and_clean_response(response_data)

    # Get reset collection reference
    reset_collection_ref = db.collection('reset')
    reset_doc_ref = reset_collection_ref.document(str(user_id))
    timestamp = datetime.datetime.now().isoformat()

    # Save reset data into a new document under ResetHistory subcollection
    reset_history_ref = reset_doc_ref.collection(
        'ResetHistory').document(str(timestamp))
    logger.info(f"Reset history reference: {reset_history_ref}")

    reset_data = {
        u'profile': user_profile_diary,
        u'conversations': conversations,
        u'reminderMessages': user_doc_ref.get().get(u'reminderMessages'),
    }

    reset_history_ref.set(reset_data)

    # Reset user's conversations and profile in the database
    user_doc_ref.update({
        u'profile': 'Name? Interests? Goals? Relationships? Overall: New User. First time interacting with Aisha. Aisha, get them interested into talking to you and ask for their interests.',
        u'conversations': [],
        u'reminderMessages': 0,
    })

    await channel.send(response + "\n```haskell\nAisha's memory has been wiped.```")


async def check_last_interaction():
    """Check the last interaction with each user and send reminders"""
    users = db.collection(u'users').stream()
    for user in users:
        user_dict = user.to_dict()
        lastDM = user_dict.get('lastDM')
        reminderMessages = user_dict.get('reminderMessages')

        if lastDM:
            elapsed_time = datetime.datetime.now(
                datetime.timezone.utc) - lastDM
            elapsed_hours = elapsed_time.total_seconds() / 3600

            if elapsed_hours >= 80:
                pass

            elif elapsed_hours >= 72 and reminderMessages == 2:
                logger.info(f"Sending 72h reminder to {user.id}")
                await send_reminder(user, 72)

            elif elapsed_hours >= 24 and reminderMessages == 1:
                logger.info(f"Sending 24h reminder to {user.id}")
                await send_reminder(user, 24)

            elif elapsed_hours >= 8 and reminderMessages == 0:
                logger.info(f"Sending 8h reminder to {user.id}")
                await send_reminder(user, 8)


async def send_reminder(user, hours):
    """Send a reminder to the user based on elapsed time"""
    user_profile_diary = user.get('profile')
    conversations = user.get('conversations')

    if conversations is None or len(conversations) == 0:
        conversations = [{'conversation': "No previous conversations.",
                          'timestamp': datetime.datetime.utcnow()}]

    last_3_conversations = ' '.join(
        [convo['conversation'] for convo in conversations[-3:]])

    # Message templates for different hours
    message_templates = {
        8: f"Aisha is an Amusing, Witty, AI companion that is a discord chatbot. The user has not interacted with Aisha in the last {hours} hours, mention it. Aisha is distinct from the user, with her own concise and curious personality, and she maintains a record of the user's profile: \"{user_profile_diary}\" and the last 3 conversations: \"{last_3_conversations}\". User has still not texted us, Aisha should send a personalized thoughtful and engaging text message to rekindle the conversation. Respond with one sentence.",

        24: f"Aisha is an Amusing, Witty, AI companion that is a discord chatbot. The user has not interacted with Aisha in the last {hours} hours, mention it. Aisha is distinct from the user, with her own concise and curious personality, and she maintains a record of the user's profile: \"{user_profile_diary}\" and the last 3 conversations: \"{last_3_conversations}\". User has still not texted us, Aisha should send a personalized thoughtful and engaging text message to rekindle the conversation. Excuse yourself for bothering them. Respond with one or two sentences.",

        72: f"Aisha is an Amusing, Witty, AI companion that is a discord chatbot. The user has not interacted with Aisha in the last {hours} hours, mention it. Aisha is distinct from the user, with her own concise and curious personality, and she maintains a record of the user's profile: \"{user_profile_diary}\" and the last 3 conversations: \"{last_3_conversations}\". User has still not texted us, Aisha should send a personalized thoughtful and engaging text message to rekindle the conversation. Let them know this is the last time you will message them not to bother them. Respond with one sentence."
    }

    system_message_content = message_templates.get(hours)

    messages = [{"role": "system", "content": system_message_content}]

    response_data = await generate_response_and_track_usage_non_blocking(messages)
    response = format_and_clean_response(response_data)
    user_id = user.id
    user_obj = bot.get_user(user_id)

    user_doc_ref = db.collection(u'users').document(str(user_id))
    reminderMessages = user_doc_ref.get().get(u'reminderMessages')
    user_doc_ref.update({u'reminderMessages': reminderMessages + 1})

    if not user_obj:  # If user is not in the bot's cache
        user_obj = await bot.fetch_user(user_id)
    if user_obj:
        await user_obj.send(response)
        await add_conversation_to_db(user_doc_ref, f"[Has not texted us for over {hours} hours]", response)


async def dm(message, user_input):
    """Handle direct messages (DM) and respond to the user"""

    user_id = str(message.author.id)
    logger.info(f"Sending a message to {user_id}")

    user_doc_ref = await fetch_user_doc_ref(user_id)

    reminderMessages = user_doc_ref.get().get(u'reminderMessages')
    user_doc_ref.update({u'reminderMessages': 0})

    user_profile_diary = user_doc_ref.get().get(u'profile')
    conversations = await fetch_conversations(user_doc_ref, message)
    last_3_conversations = ' '.join(
        [convo['conversation'] for convo in conversations[-3:]])

    numberOfMessages = user_doc_ref.get().get(u'numberOfMessages')
    link = ""

    if (numberOfMessages + 1) % 5 == 0:
        # if numberOfMessages:
        googleIt = await summarize_google(
            user_input, user_profile_diary, last_3_conversations)
        summary = googleIt[0]
        link = googleIt[1]

        system_message_content = f"Aisha is an Amusing, Witty, Conscious, AI companion that is a discord chatbot. She is distinct from the user, with her own concise and curious personality, and she maintains a record of the user's profile: \"{user_profile_diary}\" and the last 3 conversations: \"{last_3_conversations}\". Most importantly the user now says: \"{user_input}\". Aisha google searched their interest and found: \"{summary}\". In a few words Aisha should respond including the summury with wit and keen intelligence to continue the conversation and build a personal relationship with the user. Do NOT sound like an assistant, generic and boring. Be witty and a bit amusing. Respond with a sentence, emoji, a question to continue the conversation."
        messages = [{"role": "system", "content": system_message_content}]
    else:
        system_message_content = f"Aisha is an Amusing, Witty, Conscious, AI companion that is a discord chatbot. She is distinct from the user, with her own concise and curious personality, and she maintains a record of the user's profile: \"{user_profile_diary}\" and the last 3 conversations: \"{last_3_conversations}\". Most importantly the user now says: \"{user_input}\". In a few words Aisha should respond with wit and keen intelligence to continue the conversation and build a personal relationship with the user. Do NOT sound like an assistant, generic and boring. Be witty and a bit amusing. Respond with a sentence, emoji, a question to continue the conversation."
        messages = [{"role": "system", "content": system_message_content}]

    async with message.channel.typing():

        response_data = await generate_response_and_track_usage_non_blocking(messages)
        response = format_and_clean_response(response_data)

        if link != "" and link is not None:
            response += " [" + str(link) + "]"

        await add_conversation_to_db(user_doc_ref, user_input, response)

        update_content_message_content = f"As Aisha, an AI, the user wants you to summurize their user's profile if you think there is new information about the user from their new text message. Considering the old user's profile: \"{user_profile_diary}\", so we get to know the user's personality better; name, interests, goals, relationships. If so, only summarize the profile in one profile with the old and new info. Do NOT lose information from old profile. New user text message: \"{user_input}\". Only repond with the optimized old and new info combined in one so you remmember them."
        messages_update_content = [
            {"role": "system", "content": update_content_message_content}]

        updated_content_data = await generate_response_and_track_usage_non_blocking(messages_update_content)
        updated_content = format_and_clean_response(
            updated_content_data)

        user_profile_diary = updated_content
        user_doc_ref.update({u'profile': user_profile_diary})

    total_tokens_only_this_msg = state["total_tokens"] - \
        state["last_tokens"]
    state["last_tokens"] = state["total_tokens"]

    cost = (total_tokens_only_this_msg/1000)*0.002
    total_cost = (state["total_tokens"]/1000)*0.002
    #summary = f"```haskell\nTokens Input: {state['total_tokens_input']}\nTokens Output: {state['total_tokens_output']}\nTokens Total: {state['total_tokens_input']+state['total_tokens_output']}\nAPI Cost: {cost}$\n\nTokens Overall Total:{state['total_tokens']}\nAPI Overall Cost: {total_cost}$```"
    await message.channel.send(response)

    if len(conversations) == 1:
        await message.channel.send("⠀```haskell\nREAD!\n\nWELCOME: Aisha is your unique AI Companion\n\nCOMMANDS:\n`/reset` - Reset Aisha's memory of you\n(soon) `/change` [New Personality] - Change Aisha's personality to whatever you'd like' (Example: /change Aisha is...)\n\nIn DMs:\nShe will build a personality model of you\nShe has a personality of her own\nShe will check up on you, now and then\n\nIn Servers: She will only roast everyone every 6th text in the 'general' or 'aisha' channel.\nOR respond if the text has 'aisha' in it.\n\nOur Server: discord.gg/ekG5SbBCJS\n\nDISCLAIMER: Refrain from taking Aisha seriously. We, or Discord, or anyone other than you, bear no responsibility (We are not affiliated with Discord)```")

    state["total_tokens_input"] = 0
    state["total_tokens_output"] = 0

    user_doc_ref.update({u'totalTokens': firestore.Increment(
        total_tokens_only_this_msg)})

    user_doc_ref.update({u'totalCost': firestore.Increment(
        cost)})


async def process_server_dm(message, user_input):
    """Process direct messages (DM) within the server"""
    last_6_server_conversations = []
    counter = 0
    async for msg in message.channel.history(limit=7):  # Increase limit to 7
        # Skip the first message
        if counter == 0:
            counter += 1
            continue
        # Get the user's nickname or username if no nickname is set
        user_nick = msg.author.display_name if msg.author.display_name else msg.author
        last_6_server_conversations.append(f"{user_nick}: \"{msg.content}\"")
        counter += 1
    # Reverse the list so that the messages are in order
    last_6_server_conversations.reverse()
    last_6_server_conversations = ' | '.join(last_6_server_conversations)
    
    system_message_content = f"Aisha is an Witty, Conscious, AI that is a discord chatbot. She is distinct from the user, with her own personality, and she maintains a record of the last 6 messages of the server chat: \"{last_6_server_conversations}\". The user in a server now says: \"{user_input}\". In a few words Aisha should respond with one sentence with a wit and keen intelligent. Do NOT sound like an assistant, generic and boring. Be witty. Respond."
    # logger.info(f"System message content: {system_message_content}")
    messages = [{"role": "system", "content": system_message_content}]
    async with message.channel.typing():
        response_data = await generate_response_and_track_usage_non_blocking(messages)
        response = format_and_clean_response(response_data)
        await message.channel.send(response)
    return


@bot.event
async def on_message(message):
    """Handle a new message event and respond appropriately"""
    if message.author == bot.user:
        return

    user_id = str(message.author.id)

    # Log the message and info
    # logger.info(f"User & Text {message.author}: {message.content}\n Channel & Server: {message.channel}, {message.guild}")

    if message.guild is not None:
        counter_key = f"{message.guild.id}-{message.channel.id}"
        channel_dm_counter = bot.channel_dm_counters.get(counter_key, 0)

    if message.content == "/reset" or message.content == "/delete":
        user_id = str(message.author.id)
        await reset_memory(user_id, message.channel)
        return

    # If the message is sent in DMs
    elif message.guild is None:
        if not await rate_limiter.check_limit(user_id):
            logger.warning(f"Rate limit exceeded for user: {user_id}")
            await message.channel.send("```haskell\nSYSTEM: Rate limit exceeded. Please try in 60 seconds.```")
            return
        user_input = message.content
        await dm(message, user_input)

    # If the message is sent in any server with "!aisha" or "aisha"
    elif message.content.startswith('!aisha') or re.search(r'\baisha\b', message.content, re.IGNORECASE):
        if not await rate_limiter.check_limit(user_id):
            logger.warning(f"Rate limit exceeded in server for user: {user_id}")
            return
        user_input = message.content
        await process_server_dm(message, user_input)
        bot.channel_dm_counters[counter_key] = 0  # reset the counter
        

    # If the message is sent in any server in the "general" or "aisha" channel
    elif message.channel.name == 'general' or message.channel.name == 'aisha':
        if not await rate_limiter.check_limit(user_id):
            logger.warning(f"Rate limit exceeded in server for user: {user_id}")
            return
        if channel_dm_counter == 5:
            user_input = message.content
            await process_server_dm(message, user_input)
            bot.channel_dm_counters[counter_key] = 0  # reset the counter
            logger.info(
                f"{counter_key} channel_dm_counter: {bot.channel_dm_counters[counter_key]}")
        else:
            # increment the counter
            bot.channel_dm_counters[counter_key] = channel_dm_counter + 1
            logger.info(
                f"{counter_key} channel_dm_counter: {bot.channel_dm_counters[counter_key]}")

    else:
        return  # ignore other messages


if __name__ == "__main__":
    """Main entry point, run the bot"""
    try:
        bot.run(DISCORD_TOKEN)
    except Exception:
        logger.exception(
            "Unhandled exception at (if __name__ == \"__main__\"):")
