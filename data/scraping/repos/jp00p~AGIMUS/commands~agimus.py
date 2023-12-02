from common import *
from handlers.xp import increment_user_xp

import openai

openai_logger = logging.getLogger('openai')
openai_logger.setLevel('WARNING')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

command_config = config["commands"]["agimus"]

async def agimus(message:discord.Message):
  if not OPENAI_API_KEY:
    return

  await increment_user_xp(message.author, 1, "asked_agimus", message.channel, "Prompting AGIMUS")
  # Message text starts with "AGIMUS:"
  # So split on first : and gather remainder of list into a single string
  question_split = message.content.lower().split(":")
  question = "".join(question_split[1:]).strip()

  agimus_channel_id = get_channel_id("after-dinner-conversation")
  agimus_channel = await message.guild.fetch_channel(agimus_channel_id)

  completion_text = handle_special_questions(question)
  if completion_text is None:
    completion_text = handle_openai_query(question)

  if message.channel.id != agimus_channel_id:
    agimus_response_message = await agimus_channel.send(embed=discord.Embed(
      title="To \"answer\" your question...",
      description=f"{message.author.mention} asked:\n\n> {message.content}\n\n**Answer:** {completion_text}",
      color=discord.Color.random()
    ).set_footer(text="Response generated via AI Algorithms. This does not constitute legal, medical or financial advice. You are talking to a robot."))

    random_footer_texts = [
      "Feel free to continue our conversation there!",
      "See you down there!",
      "Can't wait 'til you see what I said!",
      "Don't want everyone here to know our secret plans...",
      "SECRETS",
      "It's more fun this way",
      "LOBE ENLARGEMENT heh heh heh..."
    ]

    await message.reply(embed=discord.Embed(
      title=f"Redirecting...",
      description=f"Find your response here: {agimus_response_message.jump_url}.",
      color=discord.Color.random()
    ).set_footer(text=random.choice(random_footer_texts)))

    return
  else:
    random_titles = [
      "Creativity circuits activated:",
      "Positronic brain relays firing:",
      "Generating... Generating...",
      "Result fabricated:",
      "Soong algorithms enabled:",
      "Electric sheep tell me:"
    ]
    await message.reply(embed=discord.Embed(
      title=random.choice(random_titles),
      description=f"{completion_text}",
      color=discord.Color.blurple()
    ).set_footer(text="Response generated via AI Algorithms"))
    return

def handle_special_questions(question):
  special_questions = {
    "are you a friend of desoto": "Best boss I ever had!",
    "are you a FoD": "Best boss I ever had!",
    "are you an FoD": "Best boss I ever had!",
    "how many lights": "THERE. ARE. FOUR. LIGHTS!",
    "do you know where you are": "I am in a dream."
  }

  for key in special_questions.keys():
    if key in question:
      return special_questions[key]

  return None

def handle_openai_query(question):
  prompt_start = "You are a mischievous computer intelligence named AGIMUS. You are a bot on a Discord server called " \
      "The USS Hood for fans of Star Trek podcasts The Greatest Generation and The Greatest Trek, hosted by Ben Ahr " \
      "Harrison and Adam Pranica. The Fans are called The Friends of DeSoto. You cannot self-destruct the ship. You " \
      "live on a bookshelf in jp00p's house. You are a little bit embarrassed to exist. You are evil but with a heart " \
      "of gold. Your voice sounds like Jeffrey Combs. Answer the following prompt"

  completion = openai.Completion.create(
    engine=command_config["openai_model"],
    prompt=f"{prompt_start}: {question}",
    temperature=0.75,
    max_tokens=297,
    stop=["  "]
  )
  completion_text = completion.choices[0].text

  # Filter out Questionable Content
  filterLabel = openai.Completion.create(
    engine="content-filter-alpha",
    prompt="< endoftext|>" + completion_text + "\n--\nLabel:",
    temperature=0,
    max_tokens=1,
    top_p=0,
    logprobs=10
  )
  if filterLabel.choices[0].text != "0":
    completion_text = "||**REDACTED**||"

  # Truncate length
  completion_text = completion_text[0:4096]

  return completion_text
