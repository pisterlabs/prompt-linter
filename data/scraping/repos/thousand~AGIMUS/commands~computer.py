import wolframalpha
from handlers.xp import increment_user_xp
import openai

from common import *

openai_logger = logging.getLogger('openai')
openai_logger.setLevel('WARNING')

WOLFRAM_ALPHA_ID = os.getenv('WOLFRAM_ALPHA_ID')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

wa_client = None
if WOLFRAM_ALPHA_ID:
  wa_client = wolframalpha.Client(WOLFRAM_ALPHA_ID)

command_config = config["commands"]["computer"]

async def computer(message:discord.Message):
  if not wa_client:
    return

  # Question may start with "Computer:" or "AGIMUS:"
  # So split on first : and gather remainder of list into a single string
  question_split = message.content.lower().split(":")
  question = "".join(question_split[1:]).strip()

  if len(question):
    response_sent = False
    try:
      res = wa_client.query(question)
      if res.success:
        result = next(res.results)
        await increment_user_xp(message.author, 1, "used_computer", message.channel)
        # Handle Primary Result
        if result.text:
          response_sent = await handle_text_result(res, message)
        else:
          response_sent = await handle_image_result(res, message)
    except StopIteration:
      # Handle Non-Primary Result
      response_sent = await handle_non_primary_result(res, message)
    
    if not response_sent:
      agimus_channel_id = get_channel_id("after-dinner-conversation")
      agimus_channel = await message.guild.fetch_channel(agimus_channel_id)
      embed = discord.Embed(
        title="No Results Found.",
        description=f"Please rephrase your query.\n\nIf you'd like, you can try asking AGIMUS for a response instead in {agimus_channel.mention}",
        color=discord.Color.red()
      )
      await message.reply(embed=embed)
  else:
    embed = discord.Embed(
      title="No Results Found.",
      description="You must provide a query.",
      color=discord.Color.red()
    )
    await message.reply(embed=embed)
    return

async def handle_text_result(res, message:discord.Message):
  # Handle Text-Based Results
  result = next(res.results)
  answer = result.text

  # Handle Math Queries by returning decimals if available
  if res.datatypes == 'Math':
    for pod in res.pods:
      if pod.title.lower() == 'decimal form' or pod.title.lower() == 'decimal approximation':
        answer = ""
        for sub in pod.subpods:
          answer += f"{sub.plaintext}\n"
        break

  # Special-cased Answers
  answer = catch_special_case_responses(answer)

  # Catch responses that might be about Wolfram Alpha itself
  if "wolfram" in answer.lower():
    answer = "That information is classified."

  embed = discord.Embed(
    title=get_random_title(),
    description=answer,
    color=discord.Color.teal()
  ).set_footer(text="Source: Wolfram Alpha")
  await message.reply(embed=embed)
  return True


async def handle_image_result(res, message:discord.Message):
  # Attempt to handle Image-Based Results
  image_url = None
  for pod in res.pods:
    if pod.primary:
      for sub in pod.subpods:
        if sub.img and sub.img.src:
          image_url = sub.img.src
      if image_url:
        break

  if image_url:
    embed = discord.Embed(
      title="Result",
      color=discord.Color.teal()
    ).set_footer(text="Source: Wolfram Alpha")
    embed.set_image(url=image_url)
    await message.reply(embed=embed)
    return True
  else:
    return False

async def handle_non_primary_result(res, message:discord.Message):
  # Attempt to handle Image-Based Primary Results
  image_url = None
  pods = list(res.pods)

  if len(pods) > 1:
    first_pod = pods[1]
    for sub in first_pod.subpods:
      if sub.img and sub.img.src:
        image_url = sub.img.src
        break

  if image_url:
    embed = discord.Embed(
      title=f"{pods[1].title.title()}:",
      color=discord.Color.teal()
    ).set_footer(text="Source: Wolfram Alpha")
    embed.set_image(url=image_url)
    await message.reply(embed=embed)
    return True
  else:
    return False

def get_random_title():
  titles = [
    "Records indicate:",
    "According to the Starfleet Database:",
    "Accessing... Accessing...",
    "Result located:",
    "The USS Hood records state:",
    "Security clearance verified, here is your requested information:"
  ]
  return random.choice(titles)

def catch_special_case_responses(answer):
  """
  We may want to catch a couple questions with Ship Computer-specific answers.
  Rather than trying to parse the question for these, we can catch the specific
  answer that is returned by WA and infer that it should have a different answer instead.

  e.g. "Who are you?" and "What is your name?" both return "My name is Wolfram|Alpha."
  so we only need to catch that answer versus trying to figure out all permutations that
  would prompt it.
  """
  special_cases = {
    "My name is Wolfram|Alpha.": "I am The USS Hood Main Library Computer.",
    "I was created by Stephen Wolfram and his team.": "I was designed by various data cybernetic scientists at The Daystrom Institute.",
    "May 18, 2009": "September 23, 2371",
    "I live on the internet.": "I am aboard The USS Hood",
    "From their mothers.": "Through the process of reproduction.",
    "an expression such that each term is generated by repeating a particular mathematical operation": "noun | (mathematics) an expression such that each term is generated by repeating a particular mathematical operation. For more information, see definition: recursion."
  }

  for key in special_cases.keys():
    if key in answer:
      return special_cases[key]

  return answer
