from discord.ext import commands
import re
import openai
from config.settings import GPT_TOKEN

model = "gpt-4"
resp_prompt = """
I have a discord bot, "FamBot", that fixes links shared in the chat. Generate a short response that the bot would reply with when posting the fixed link. Use a personality for the bot that's akin to a internet commenter and zoomer. Just include the response without quotes. Don't include placeholder for the link like [fixed link].
"""
class VideoLinkEmbeds(commands.Cog):
    
    def __init__(self, bot):
        self.bot = bot
        openai.api_key = GPT_TOKEN
        # substrings to be inserted in url for fixing embeds
        self.sources = {
            "instagram.com": "dd",
            "tiktok.com": "vx",
            "twitter.com": "vx",
            "x.com": "fixv",
        }

    def find_valid_url(self, message):
        url_pattern = re.compile(r'https?://(?:www\.)?(\w+\.com)(?!\.com)')
        # find all URLs in the msg content
        found_url = url_pattern.search(message)
        if found_url:
            # extract the domain part from the URL
            domain = found_url.group(1)
            # compare the extracted domain against base urls
            if domain in self.sources:
                return found_url.group(0)
            return None

    @commands.Cog.listener()
    async def on_message(self, msg):
        ### hardcoded responses if gpt doesn't work out
        # responses = {
        #     "I gotchu, Fam",
        #     "Oopsie daisy, someone was lazy, but I'll fix it cuz I'm cRaZy lmao",
        #     "Well that didn't work. Here ya go.",
        #     "smh, I'll fix it",
        #     "Let's just go ahead and fix up that link real quick",
        #     "They really need to fix this already, but I gotchu",
        #     "That's another nickle in the 'Fambot is the GOAT' jar",
        #     "And some of yall have me blocked when I'm out here being helpful, smh"
        # }

        # Regex pattern to match the desired URLs and ignore any with ".com" in the path/query
        # god this is getting long pls work
        url_pattern = re.compile(r'https?://(?:www\.)?([\w-]+\.com)(?!.*?\.com)(/[^?#\s]*)?(\?[^#\s]*)?(#[^\s]*)?')

        if msg.author == self.bot.user:
            return
        
        found_url = url_pattern.search(msg.content)

        # add substring into url
        if found_url:
            url = found_url.group()

            # detect if embed exists already and has a video element
            if (len(msg.embeds) > 0 and msg.embeds[0].video) \
                or (len(msg.embeds) > 0 and "twitter.com" in url):
                return

            # gpt generated response
            gpt_response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": resp_prompt}],
                timeout=30,
                user=msg.author.name
            )
            print(gpt_response)
            fambot_response = gpt_response.choices[0].message.content
            print("------ ai response generated ------")
            print(f"- {fambot_response}")

            for base, modification in self.sources.items():
                # add modification to front of base url
                if base in url and f"{modification}{base}" not in url:
                    embed_url = url.replace(base, f"{modification}{base}")
                    await msg.channel.typing()
                    await msg.channel.send(f'{fambot_response}\n{embed_url}')
                    break # only fix a single link in a single message

async def setup(bot):
    await bot.add_cog(VideoLinkEmbeds(bot))
