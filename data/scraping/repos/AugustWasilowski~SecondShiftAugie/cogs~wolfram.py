import logging

from langchain.agents import load_tools, initialize_agent
from nextcord.ext import commands

from cogs.status import working, wait_for_orders
from langchain import OpenAI

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def execute_wolfram_alpha(ctx, arg):
    """Executes Wolfram Alpha"""
    try:
        wolf_llm = OpenAI(temperature=0)
        tool_names = ["wolfram-alpha"]
        tools = load_tools(tool_names)
        agent = initialize_agent(
            tools, wolf_llm, agent="zero-shot-react-description", verbose=True
        )
        result = agent.run(arg)
        await ctx.send(result)
        # await self.generate_voice_sample(result)
    except Exception as e:
        logger.error(f"General error in Wolfram: {e}")
        await ctx.send(f"Error in Wolfram: {e}.")


def setup(bot: commands.Bot):
    bot.add_cog(WolframAlphaCog(bot))


class WolframAlphaCog(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.is_busy = False
        # self.status_cog = self.bot.get_cog(self, 'status')

    @commands.command()
    async def wolf(self, ctx, *, arg):
        """ "Sets status then executes Wolfram Alpha query"""
        if not self.is_busy:
            await working(self.bot)
            await execute_wolfram_alpha(ctx, arg)
            # await self.play_latest_voice_sample()
            await wait_for_orders(self.bot)
        # else:
        # await self.generate_voice_sample("I'm busy at the moment. Please wait.")
        # await self.play_latest_voice_sample()

    async def set_busy(self, message):
        await self.status_cog.get_commands(self)
