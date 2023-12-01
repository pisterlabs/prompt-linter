"""
from langchain.utilities import BingSearchAPIWrapper

search = BingSearchAPIWrapper(k=10) # type: ignore

print(search.run('How many episodes in picard season 3'))

"""

import asyncio
from utils import tokenize_text
from web_extractor import WebExtractor

def tokenizer(text):
    return tokenize_text(text)

async def main():
    extractor = WebExtractor()

    raw_results = await extractor.extract_text('https://en.wikipedia.org/wiki/Star_Trek:_Picard_(season_3)', tokenizer=tokenizer, tokens_per_chunk=3000)
    i = 0
    for result in raw_results:
        i += 1
        print("RESULT #" + str(i) + "\n" + result + "\n\n")

if __name__ == "__main__":
    asyncio.run(main())
