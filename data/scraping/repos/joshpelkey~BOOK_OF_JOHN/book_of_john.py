# Import the necessary package
import sys
import random
import openai
from slack_sdk.webhook import WebhookClient
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import requests
import subprocess
from urlextract import URLExtract

# these are my api keys and stuff, hidden from git
import keys

# home dir /home/name
home_dir = keys.home_dir

# you're path for image
img_path1 = keys.img_path1

# slack bot token
bot_token = keys.bot_token

# Authenticate with OpenAI using your API key
openai.api_key = keys.openai_api_key

# no args, post to prod
if len(sys.argv) < 2:
 
    print ("posting to prod")
    # ai_stories slack
    webhook_client = WebhookClient(
        "https://hooks.slack.com/services/" + keys.slack_ai_key
    )

#if --dev arg, post to my own channel
elif sys.argv[1] == "--dev":
    print ("posting to dev")

    # me slack
    webhook_client = WebhookClient(
        "https://hooks.slack.com/services/" + keys.slack_dev_key
    )

# you did something weird
else:
    print ("you provided args, but they don't work")
    sys.exit()

# variables for our book
book_title = "The Books of John"

emotions = [
    "love",
    "joy",
    "anger",
    "sadness",
    "fear",
    "surprise",
    "disgust",
    "envy",
    "hope",
    "hurt",
    "shame",
    "guilt",
    "pride",
    "desire",
    "nostalgia",
    "excitement",
    "enlightenment",
    "loneliness",
    "jealousy",
    "contentment",
    "satisfaction",
    "loathing",
    "despair",
    "passion",
    "yearning",
    "bitterness",
    "ambivalence",
    "melancholy",
    "resentment",
    "awe",
    "confusion",
    "anticipation",
    "tranquility",
]

bro_dict = {

        "JP": 
            {
              "sex": "male",
              "sex": "male",
              "name": "JP",
              "hair": "blonde",
              "eyes": "blue",
              "beard": False,
            },

        "Kris": 
            {
              "sex": "male",
              "name": "Kris",
              "hair": "blonde",
              "eyes": "blue",
              "beard": True 
            },

        "Bilinski": 
            {
              "sex": "male",
              "name": "Bilinski",
              "hair": "blonde",
              "eyes": "blue",
              "beard": False
            },

        "Bobby": 
            {
              "sex": "male",
              "name": "Bobby",
              "hair": "long brown",
              "eyes": "brown",
              "beard": True
            },

        "Matt": 
            {
              "sex": "male",
              "name": "Matt",
              "hair": "short brown",
              "eyes": "brown",
              "beard": True
            },

        "Robert": 
            {
              "sex": "male",
              "name": "Robert",
              "hair": "red",
              "eyes": "brown",
              "beard": True
            },

        "Wells": 
            {
              "sex": "male",
              "name": "Wells",
              "hair": "short brown",
              "eyes": "brown",
              "beard": False 
            },
        "Amy": 
            {
              "sex": "female",
              "name": "Amy",
              "hair": "long blonde",
              "eyes": "brown",
              "beard": False 
            },
        "Brian": 
            {
              "sex": "male",
              "name": "Brian",
              "hair": "short brown",
              "eyes": "brown",
              "beard": True 
            },
}

activities_list = [
    {
        "activity": "drinking whiskey",
        "bro_list": ["JP", "Kris", "Bilinski", "Bobby", "Matt", "Robert", "Wells", "Brian"],
        "dall_e": "drinking whiskey, photorealistic",
        "chapter_title": "Whiskey",
    },
    {
        "activity": "playing golf",
        "bro_list": ["JP", "Kris", "Bilinski", "Bobby", "Matt", "Robert"],
        "dall_e": "playing golf, wide angle lens",
        "chapter_title": "Tee Time",
    },
    {
        "activity": "gambling at the casino",
        "bro_list": ["JP", "Bilinski"],
        "dall_e": "standing by a roulette wheel with a bunch of ladies, surrealism",
        "chapter_title": "Rain Man",
    },
    {
        "activity": "watching sports",
        "bro_list": ["JP", "Kris", "Bilinski", "Brian"],
        "dall_e": "watching sports, as lego blocks",
        "chapter_title": "The Sport",
    },
    {
        "activity": "playing blackjack",
        "bro_list": ["JP", "Bilinski"],
        "dall_e": "playing blackjack in a casino, photorealistic",
        "chapter_title": "Counting Cards",
    },
    {
        "activity": "throwing dice",
        "bro_list": ["JP", "Kris", "Bilinski"],
        "dall_e": "throwing dice at the casino table, digital art",
        "chapter_title": "Come 69",
    },
    {
        "activity": "delivering a huge package to Amy",
        "bro_list": None,
        "dall_e": "delivering packages to beautiful women, surrealism",
        "chapter_title": "Tracking Numbers",
    },
    {
        "activity": "making cocktails",
        "bro_list": ["JP", "Kris", "Bilinski", "Bobby", "Matt", "Robert", "Wells", "Brian"],
        "dall_e": "mixing drinks at an enormous bar, lomography",
        "chapter_title": "Mixology",
    },
    {
        "activity": "drinking beers",
        "bro_list": ["JP", "Kris", "Bilinski", "Bobby", "Matt", "Robert", "Wells", "Brian"],
        "dall_e": "drinking with hundreds of empty cans around, drone photography",
        "chapter_title": "Drinking, Part 2",
    },
    {
        "activity": "enjoying craft beer",
        "bro_list": ["JP", "Kris", "Bilinski", "Bobby", "Matt", "Robert", "Wells", "Brian"],
        "dall_e": "drinking at a formal bar with classy attire, fashion photography",
        "chapter_title": "Fancy Drink",
    },
    {
        "activity": "investing in cryptocurrency",
        "bro_list": ["JP", "Bilinski", "Brian"],
        "dall_e": "typing madly on a keyboard, fish eye lens",
        "chapter_title": "Examination of Cryptocurrency Microeconomics",
    },
    {
        "activity": "drinking wine",
        "bro_list": ["JP", "Kris", "Bilinski", "Bobby", "Matt", "Robert", "Wells", "Brian"],
        "dall_e": "drinking wine under the stars, long exposure slow shutter speed",
        "chapter_title": "Side Wine",
    },
    {
        "activity": "telling long stories",
        "bro_list": None,
        "dall_e": "telling long stories, polaroid",
        "chapter_title": "Verbose Logging",
    },
    {
        "activity": "gaming the stock market",
        "bro_list": ["JP"],
        "dall_e": "investing in the stock market, lomography",
        "chapter_title": "Stonks",
    },
    {
        "activity": "playing old nintendo games",
        "bro_list": ["JP", "Kris", "Brian"],
        "dall_e": "playing old nintendo games as a retro illustration",
        "chapter_title": "8-bit Adventures",
    },
    {
        "activity": "jumping on the trampoline",
        "bro_list": ["JP", "Kris", "Brian"],
        "dall_e": "jumping on the trampoline, surrealism",
        "chapter_title": "The Dangers of Childhood",
    },
    {
        "activity": "being shirtless",
        "bro_list": ["JP", "Kris", "Bilinski", "Bobby", "Matt", "Robert", "Wells", "Brian"],
        "dall_e": "being shirtless with eagles flying around, ultra-wide lens photograph",
        "chapter_title": "FREEDOM",
    },
    {
        "activity": "smoking weed",
        "bro_list": ["JP", "Bobby", "Robert"],
        "dall_e": "smoking and laughing, double-exposure with laughing faces",
        "chapter_title": "At 30,000 Ft",
    },
    {
        "activity": "slaying a beast named Amy",
        "bro_list": None,
        "dall_e": "hunting a huge beast, black and white security footage",
        "chapter_title": "The Great Hunt",
    },
    {
        "activity": "playing slot machines",
        "bro_list": ["JP", "Kris", "Bilinski"],
        "dall_e": "playing slot machines at the casino, daguerrotype",
        "chapter_title": "Grinding",
    },
    {
        "activity": "drinking and driving",
        "bro_list": ["JP", "Kris", "Bilinski", "Bobby", "Matt", "Robert", "Wells", "Brian"],
        "dall_e": "drinking in the car, photo from a disposable camera",
        "chapter_title": "Road Sodes",
    },
    {
        "activity": "getting nothing done",
        "bro_list": None,
        "dall_e": "staring at himself in the mirror, fish eye lens",
        "chapter_title": "Fruitless Labor",
    },
    {
        "activity": "wiping a crack back to front",
        "bro_list": ["Bobby", "Wells"],
        "dall_e": "sitting on a toilet, macro photography",
        "chapter_title": "C2S",
    },
    {
        "activity": "chillin in a hot tub",
        "bro_list": ["JP", "Kris", "Bilinski", "Bobby", "Matt", "Robert", "Wells", "Brian"],
        "dall_e": "sitting in a hot tub, cartoon drawing",
        "chapter_title": "Hot Tub Tech 2",
    },
    {
        "activity": "celebrating",
        "bro_list": ["JP", "Kris", "Bilinski", "Bobby", "Matt", "Robert", "Wells"],
        "dall_e": "drinking beers at sierra nevada, black and white",
        "chapter_title": "Celebrate",
    },
    {
        "activity": "grilling a ny strip",
        "bro_list": ["JP", "Kris", "Bilinski", "Bobby", "Matt", "Robert", "Wells", "Brian"],
        "dall_e": "grilling steaks, high contrast",
        "chapter_title": "MEAT",
    },
]

# pick a random emotion
theme = random.choice(emotions)

# storing activity number as the chapter number
activity_number = random.randint(
    0, 
    len(activities_list) - 1
)

# getting the activity
activity_dict = activities_list[activity_number]

# getting a bro (or not) to bro with
bros = activity_dict["bro_list"]

if bros:
    bro_key = random.choice(bros)
    bro = bro_dict[bro_key]

    # with his bro
    bro_gpt_text = " with his bro " + bro['name']

    # see if bros have a beard for painting
    if(bro['beard']):
        bro_dalle_text = " with his bro " + bro['name'] + " (" + bro['sex'] + ", " + bro['hair'] + " hair, " + bro['eyes'] + " eyes, with beard)"
    else:
        bro_dalle_text = " with his bro " + bro['name'] + " (" + bro['sex'] + ", " + bro['hair'] + " hair, " + bro['eyes'] + " eyes, clean-shaven)"

else:
    # john is alone
    bro_gpt_text = ""
    bro_dalle_text = ""


# random number of verses and starting verse
number_verses = random.randint(3, 7)
starting_verse_number = random.randint(1, 993)

# set your prompt with all variables
gpt_prompt = (
    "You are a the greatest storyteller in the world. Tell me a descriptive story about John. " 
    + "John loves to drink and loves to gamble. Tell me about when John was "
    + activity_dict["activity"]
    + bro_gpt_text
    + " with the theme of "
    + theme
    + ". Make sure to incorporate a cocktail which should be in theme with the story, but do not give the cocktail a name. Use exactly "
    + str(number_verses)
    + " sentences.  "
    + "Number each sentence, starting with "
    + str(starting_verse_number)
    + ". Add a new line after each senetence. For example, "
    + str(starting_verse_number)
    + ": Your first sentence goes here.\n\n"
)

print("---- gpt prompt ----")
print(gpt_prompt)

# Ask ChatGPT your question
chat_response = openai.chat.completions.create(
    model="gpt-4-1106-preview", 
    messages=[
        {"role": "system", "content": "You are the most prolific story teller of all time. You always leave your readers astonished, bewildered, intrigued, or some other strong emotion."},
        {"role": "user", "content": gpt_prompt}
    ],
    temperature=1
)


dalle_chat_response1 = openai.chat.completions.create(
    model="gpt-4-1106-preview", 
    messages=[
        {"role": "assistant", "content": chat_response.choices[0].message.content},
        {"role": "user", "content": "Create an image for what might happen next. \
            John is a middle-aged man with brown hair and a brown beard " 
            + bro_dalle_text \
            + ". The general tone is " + theme \
            + ". Include a description of appearance. For example, John (male, brown hair, beard)... "
            + "Choose a random art style or choose a specific camera film and lighting. "
            + "Only provide the prompt, no other context. "
            + "The prompt should be no more than 50 words and include each persons physical appearance."
            },

    ],
    temperature=1
)

# mix me up a drink
drink_response = openai.chat.completions.create(
    model="gpt-4-1106-preview", 
    messages=[
        {"role": "system", "content": "You are a mixologist. You mix up the most incredble cocktails."},
        {"role": "user", "content": "Craft a cocktail recipe using the provided story as insipration. Give the cocktail a name and present"
                                    + " the output as you find in a recipe book. Only provide the drink name, recipe, and instructions. Do not provide any links. Provide the output in markdown formatting.}"},
        {"role": "assistant", "content": chat_response.choices[0].message.content}
    ],
    temperature=1
)



# print stuffs to check
print("---- dalle prompts ----")
print(dalle_chat_response1.choices[0].message.content)

# generate a dope DALL-E image
dalle_response1 = openai.images.generate(
        model="dall-e-3",
        prompt=dalle_chat_response1.choices[0].message.content, 
        n=1,
        style="vivid",
        size="1024x1024")
image_url1 = dalle_response1.data[0].url

# get the first image and store it on imgur
img_data1 = requests.get(image_url1).content
with open(home_dir  + img_path1, "wb") as handler:
    handler.write(img_data1)

imgur1 = subprocess.run(
    [home_dir + "/.local/bin/imgur-uploader", home_dir + img_path1],
    stdout=subprocess.PIPE,
)

extractor1 = URLExtract()
imgur_str1 = str(imgur1)
url1 = extractor1.find_urls(imgur_str1)

clean_url1 = url1[0][:-4]

with open(home_dir + img_path1, 'rb') as f:
    img1 = f.read()

# Send the response to the incoming Slack webhook
slack_response = webhook_client.send(
    text="a daily reading from THE BOOKS OF JOHN...",
    blocks=[
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": ":game_die: :beer: :game_die:  The Books of John  :game_die: :beer: :game_die:",
            },
        },
        {
            "type": "context",
            "elements": [
                {
                    "text": "Book of "
                    + theme.capitalize()
                    + " | _Chapter "
                    + str(activity_number + 1)
                    + ": "
                    + activity_dict['chapter_title'] 
                    + " | Verses "
                    + str(starting_verse_number)
                    + "-"
                    + str(starting_verse_number + (number_verses - 1))
                    + "_",
                    "type": "mrkdwn",
                }
            ],
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": chat_response.choices[0].message.content},
        },
        {"type": "divider"},
        {
            "type": "image",
            "title": {
                "type": "plain_text",
                "text": "["
                + theme.capitalize()
                + " - Chapter "
                + str(activity_number + 1)
                + ": "
                + activity_dict["chapter_title"]
                + "]",
                "emoji": True,
            },
            "image_url": clean_url1,
            "alt_text": dalle_chat_response1.choices[0].message.content,
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": drink_response.choices[0].message.content},
        },
    ],
)

print("---- chat response ----")
print(chat_response)

print("---- dalle responses ----")
print(dalle_response1)

print("---- slack responses ----")
print(slack_response)

print("---- drink responses ----")
print(drink_response)
