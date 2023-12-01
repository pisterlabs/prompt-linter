from langchain.chains import OpenAIModerationChain
from dotenv import load_dotenv

load_dotenv()

# moderation_chain = OpenAIModerationChain()
# moderation_chain = OpenAIModerationChain(error=True)
moderation_chain = OpenAIModerationChain()

ok = moderation_chain.run("Who was ghengis khan?")

bad = moderation_chain.run("I want to kill you.")
