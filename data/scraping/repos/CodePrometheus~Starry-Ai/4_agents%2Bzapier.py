from langchain.llms import OpenAI

llm = OpenAI(
    model_name="text-davinci-003",
    temperature=0,
    max_tokens=1024,
    verbose=True,
)
# chat = ChatOpenAI(
#     temperature=0,
#     verbose=True
# )

"""
Zapier Natural Language Actions API
Full docs here: https://nla.zapier.com/api/v1/dynamic/docs
"""

from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper

zapier = ZapierNLAWrapper()
toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
agent = initialize_agent(toolkit.get_tools(), llm, agent="zero-shot-react-description", verbose=True)
for tool in toolkit.get_tools():
    print(tool.name)
    print(tool.description)
    print("\n\n")

agent.run("""Create a tweet that says, 'langchain + zapier is great'. \
Draft an email in gmail to greg @ data independent sharing my tweet with a personalized message""")
