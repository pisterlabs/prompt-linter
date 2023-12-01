import logging

from langchain import OpenAI
from langchain.agents import load_tools, initialize_agent
from nextcord.ext import commands

from cogs.status import working, wait_for_orders

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup(bot: commands.Bot):
    bot.add_cog(SerpCog(bot))


class SerpCog(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.is_busy = False

    @commands.command()
    async def qq(self, ctx, *, arg):
        """Sets status and executes SerapApi"""
        if not self.is_busy:
            await working(self.bot)
            await self.exe_serpapi(ctx, arg)
            #     await play_latest_voice_sample()
            await wait_for_orders(self.bot)

    async def exe_serpapi(self, ctx, arg):
        """Executes SerapApi"""
        try:
            llm = OpenAI(temperature=0)
            tool_names = ["serpapi", "wikipedia", "wolfram-alpha"]
            tools = load_tools(tool_names)

            agent = initialize_agent(
                tools, llm, agent="zero-shot-react-description", verbose=True
            )
            result = agent.run(arg)
            await ctx.send(result)
        #    await generate_voice_sample(result)
        except Exception as e:
            logger.error(f"General error in Serpapi: {e}")
            await ctx.send(f"Error in Serapi: {e}.")
