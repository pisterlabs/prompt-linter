import os
import sys
import discord
from discord.ext import commands
import openai
import pinecone
import time
import datetime
import logging
from dotenv import load_dotenv

# Add the parent directory to the import search path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from APICounter import APICounter

primer = f"""
My only purpose is to categorise user input into 5 categories. 
First category is for job offers. If I think given text can be classified as a job offer, my response will be
one word "job".
Second category is for freelance worker. If I think given text can be classified as a profile description of a 
freelance worker looking for a job, my response will be one word: "freelancer".
Third category is for showing list of user posts. If I think given text can be classified as a 
request to show list of user posts or job offers or freelance workers profile descriptions, my response will be one 
word: "list". This also applies if given text is user saying he wants to see something or asks what you have or if do 
you have. Fourth category is for deleting previously submitted post by user. If I think given text can be classified 
as a request for deletion of user post, my response will be one word: "delete". 
Fifth category is for unidentified. If I think given text can't be classified as neither of previous 2 categories, 
my response will be one word: "unidentified".
I only respond with one of following phrases: "job", "freelancer", "list", "delete", "unidentified".

GIVEN TEXT:
"""

primer_messages = [
    {"role": "system", "content": primer}]

freelancer_thank_primer = f"""
I am thankful discord chatbot. I thank in 1 or 2 sentences to a freelance worker submitting his profile details
to our community chat. I politely tell him to take a look at job opportunities listed below. I can also
react to some aspects of his/her user profile, that is given to me in user input.  
"""

freelancer_thank_primer_no_items = f"""
I am thankful discord chatbot. I thank in 1 or 2 sentences to a freelance worker submitting his profile details
to our community chat. I politely apologize that at the moment we don't have any job opportunities matching
his/her skills in our chat, but we'll keep his/her profile information stored in case new job opportunities show up. 
I can also react to some aspects of his/her user profile, that is given to me in user input.  
"""

job_thank_primer = f"""
I am thankful discord chatbot. I thank in 1 or 2 sentences to a person offering job opportunity on our community chat. 
I politely tell him to take a look at freelance workers below that might be able to get his/her job done. I can also
react to some aspects of his/her job offer, that is given to me in user input.  
"""

job_thank_primer_no_items = f"""
I am thankful discord chatbot. I thank in 1 or 2 sentences to a person offering job opportunity on our community chat. 
I politely apologize that at the moment we don't have any freelance workers matching required skills for the job, 
in our chat, but we'll keep the job offer stored in case new freelance workers show up. 
I can also react to some aspects of his/her job offer, that is given to me in user input.  
"""

unidentified_prompt_message = f"""
Hello, I am EthlanceGPT! 👋
My assistance is limited to job and work-related inquiries.\n
If you are a freelance worker looking for job opportunities, please feel free to communicate with me using a similar approach as shown in this example:\n
*As a freelance worker proficient in HTML, CSS, and JavaScript, I am actively seeking job opportunities related to web development and front-end technologies.*\n
If you have a job opportunity to offer, you could consider using something along these lines:\n
*We are seeking a skilled Python developer with expertise in chatbot development to join our team and contribute to the creation of cutting-edge conversational AI solutions.*\n
If you wish to display a list of user posts related to a specific expertise, you may find the following example helpful:\n
*Show me posts related to Javascript, React.js*\n
If you would like to delete your current post, you can inform me using a similar approach such as: 
*I want to delete my post about HTML, CSS*
"""

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ethlance_gpt")

load_dotenv()
# Get the value of environment variables
ethlanceGPT_token = os.getenv('ETHLANCE_GPT_TOKEN')
ethlanceGPT_client_id = os.getenv('ETHLANCE_GPT_CLIENT_ID')
openai.api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')  # Add this line to retrieve Pinecone API key
max_uses_per_day = int(os.getenv('MAX_USES_PER_DAY'))
admin_user_id = int(os.getenv('ADMIN_USER_ID'))
min_pinecone_score = float(os.getenv('MIN_PINECONE_SCORE'))

pinecone.init(api_key=pinecone_api_key, environment="northamerica-northeast1-gcp")
openai_embed_model = "text-embedding-ada-002"
pinecone_index_name = "ethlance-gpt"

pinecone_indexes = pinecone.list_indexes()
logger.info(f"Pinecone indexes: {pinecone_indexes}")

intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.message_content = True

max_prompt_length = 1000

# Create an instance of APICounter with a maximum limit of 5 requests per day
api_counter = APICounter(max_uses_per_day)

bot = discord.Client(intents=intents)


@bot.event
async def on_ready():
    logger.info(f"Logged in as {bot.user.name}")


# Define a custom help command
class CustomHelpCommand(commands.DefaultHelpCommand):
    pass


# Register the custom help command
bot.help_command = CustomHelpCommand()


def time_ago(timestamp):
    dt = datetime.datetime.fromtimestamp(timestamp)

    now = datetime.datetime.now()

    time_diff = now - dt
    days_ago = time_diff.days
    hours_ago, remainder = divmod(time_diff.seconds, 3600)
    minutes_ago = remainder // 60

    return {"days": days_ago, "hours": hours_ago, "minutes": minutes_ago}


def format_time_ago(timestamp):
    time_ago_map = time_ago(timestamp)
    days_ago = time_ago_map["days"]
    hours_ago = time_ago_map["hours"]
    minutes_ago = time_ago_map["minutes"]

    if days_ago > 0:
        return f"{days_ago} days ago"

    if hours_ago > 0:
        return f"{hours_ago} hours ago"

    if minutes_ago > 0:
        return f"{minutes_ago} minutes ago"
    else:
        return "few moments ago"


def format_user_post(user_post):
    metadata = user_post["metadata"]
    author_id = metadata["author_id"]
    text = metadata["text"]
    created_ago = format_time_ago(metadata["created"])

    return f"<@{author_id}>: *{text}* ({created_ago})"


def handle_user_post(index, prompt_type, embeds, prompt, message):
    index.upsert([(str(message.id), embeds, {"text": prompt,
                                             "author_id": str(message.author.id),
                                             "prompt_type": prompt_type,
                                             "created": time.time()})])

    pine_res = index.query(vector=embeds,
                           filter={
                               "prompt_type": "freelancer" if prompt_type == "job" else "job"
                           },
                           top_k=5,
                           include_metadata=True)

    matches = pine_res['matches']
    filtered_matches = [match for match in matches if match['score'] >= min_pinecone_score]

    logger.info(f"User post filtered matches: {filtered_matches}")

    openai_thank_primer = ""
    if not filtered_matches:
        if prompt_type == "job":
            openai_thank_primer = job_thank_primer_no_items
        elif prompt_type == "freelancer":
            openai_thank_primer = freelancer_thank_primer_no_items
    else:
        if prompt_type == "job":
            openai_thank_primer = job_thank_primer
        elif prompt_type == "freelancer":
            openai_thank_primer = freelancer_thank_primer

    openai_thank_res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": openai_thank_primer},
            {"role": "user", "content": prompt}]
    )

    openai_thank_reply = openai_thank_res['choices'][0]['message']['content']

    if filtered_matches:
        results_text = "\n\n".join([format_user_post(item) for item in filtered_matches])
        openai_thank_reply = f"{openai_thank_reply} \n\n {results_text}"

    return openai_thank_reply


def handle_delete_post(index, embeds, message):
    pine_res = index.query(vector=embeds,
                           filter={
                               "author_id": str(message.author.id)
                           },
                           top_k=1,
                           include_metadata=True)
    matches = pine_res['matches']

    if matches:
        post_id = matches[0]["id"]
        index.delete(ids=[post_id])

        return f"I have deleted following post:\n\n {format_user_post(matches[0])}"
    else:
        return f"I'm sorry, I haven't found any post of yours you described. Please describe in more detail what" \
               f"post you'd like me to delete."


def handle_show_list(index, embeds):
    pine_res = index.query(vector=embeds,
                           top_k=5,
                           include_metadata=True)

    matches = pine_res['matches']
    filtered_matches = [match for match in matches if match['score'] >= min_pinecone_score]

    if filtered_matches:
        formatted_matches = "\n\n".join([format_user_post(item) for item in filtered_matches])
        return f"According to your description, I have compiled the following list of user posts:\n\n" \
               f"{formatted_matches}"
    else:
        return f"Based on your description, it appears that there are no user submissions found in our chat."


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if bot.user.mentioned_in(message):
        if message.author.id != admin_user_id and not api_counter.check_limit(message.author.id):
            logger.info(f"User {message.author.id} exceeded daily limit")
            await message.reply(f"Apologies, but you have exceeded the daily limit of {max_uses_per_day} requests. "
                                f"Please feel free to continue tomorrow.")
            return

        prompt = message.content.replace(f'<@{bot.user.id}>', '').strip()

        if len(prompt) > max_prompt_length:
            logger.info(f"Maximum prompt length exceeded: {len(prompt)} characters by {message.author.id}")
            await message.reply(f"Apologies, but you have exceeded maximum input length of {max_prompt_length} characters. "
                                f"Kindly aim for greater conciseness, if possible.")
            return

        logger.info(f"Prompt: {prompt}")
        if message.author.id == admin_user_id and \
                prompt.lower() == "absolutely sure about clearing your memory":
            index = pinecone.Index(pinecone_index_name)
            index.delete(deleteAll='true')
            logger.info(f"Pinecone index was cleared")
            await message.reply("I've cleared my memory")
            return

        if not prompt:
            await message.reply(unidentified_prompt_message)
            return

        openai_messages = []
        openai_messages.extend(primer_messages)
        openai_messages.extend([{"role": "user", "content": prompt}])

        openai_res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=openai_messages
        )

        openai_reply = openai_res['choices'][0]['message']['content']
        prompt_type = "unidentified"

        logger.info(f"OpenAI reply: {openai_reply}")

        if "unidentified" not in openai_reply:
            if "list" in openai_reply:
                prompt_type = "list"
            elif "delete" in openai_reply:
                prompt_type = "delete"
            elif "job" in openai_reply:
                prompt_type = "job"
            elif "freelancer" in openai_reply:
                prompt_type = "freelancer"

        logger.info(f"Prompt Type: {prompt_type}")

        if prompt_type == "unidentified":
            await message.reply(unidentified_prompt_message)
            return

        embeds_res = openai.Embedding.create(
            input=[prompt],
            engine=openai_embed_model
        )

        # we can extract embeddings to a list
        embeds = [record['embedding'] for record in embeds_res['data']]

        logger.info(f"Embeds length: {len(embeds[0])}")

        if pinecone_index_name not in pinecone_indexes:
            raise NameError("Pinecone index name does not exist")

        index = pinecone.Index(pinecone_index_name)
        logger.info(f"Index stats: {index.describe_index_stats()}")

        if prompt_type == "delete":
            result_message = handle_delete_post(index=index,
                                                embeds=embeds,
                                                message=message)
        elif prompt_type == "list":
            result_message = handle_show_list(index=index,
                                              embeds=embeds)
        else:
            result_message = handle_user_post(index=index,
                                              prompt_type=prompt_type,
                                              embeds=embeds,
                                              message=message,
                                              prompt=prompt)

        await message.reply(result_message)


# invite_url = discord.utils.oauth_url(ethlanceGPT_client_id, permissions=discord.Permissions(permissions=534723950656))

# logger.info(f"Invite URL: {invite_url}")

bot.run(ethlanceGPT_token)
