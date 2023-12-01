import time
import asyncio
from langchain.chat_models import ChatOpenAI
import pandas as pd


from async_chain import CharacteristicsChain, SummaryChain

llm = ChatOpenAI(
    temperature=0.0,
    request_timeout=15,
    model_name="gpt-3.5-turbo",
)

df = pd.read_csv('wine_subset.csv')

# Issue Chain
s = time.perf_counter()
summary_chain = SummaryChain(llm=llm,df=df)
asyncio.run(summary_chain.generate_concurrently())
elapsed = time.perf_counter() - s
print("\033[1m" + f"Summary Chain (Async) executed in {elapsed:0.2f} seconds." + "\033[0m")

# Characteristics Chain
s = time.perf_counter()
characteristics_chain = CharacteristicsChain(llm=llm,df=df)
asyncio.run(characteristics_chain.generate_concurrently())
elapsed = time.perf_counter() - s
print("\033[1m" + f"Characteristics Chain (Async) executed in {elapsed:0.2f} seconds." + "\033[0m")

df.to_csv('checkpoint.csv')
