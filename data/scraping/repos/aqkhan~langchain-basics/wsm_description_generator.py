import sys
if len(sys.argv) > 1:
    print(f"SKU: {sys.argv[1]}")
else:
    print("Missing required argument (SKU)")
    sys.exit()
if len(sys.argv) > 2:
    print(f"Brand: {sys.argv[2]}")
else:
    print("Missing required argument (Brand)")
    sys.exit()

sku = sys.argv[1]
brand = sys.argv[2]

from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()


os.environ['SERPAPI_API_KEY'] = os.getenv("SERPAPI_API_KEY")
open_ai_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    temperature=0.7,
    # temperature=0,
    model_name="gpt-3.5-turbo",
    openai_api_key=open_ai_key
)

# Describe tools and agent

tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(tools, llm, agent="zero-shot-react-description")

template = ChatPromptTemplate.from_messages([
    ("system", "You're an automative expert. You can generate description about an automotive product based on automotive product's SKU and the brand. You need to write at least 300 words. You can search the internet and for broader context search for the automotive product's brand and the sku. The title and the description of the product should be separated by a \n. In the description of the product, try to include the compatibility of the proudct with the model years of the vehicle and information such VIN if you can find it. Please don't include the price in description. Do not include product SKU inside the product descirption."),
    ("human", "{brand} {sku}")
])

messages = template.format_messages(
    brand=brand,
    sku=sku
)

resp = agent.run(messages)
print(resp)