import anthropic
import sys
import asyncio
import time
import os

ANTHROPIC_API_KEY = 'YOUR-KEY'

async def summarize_chunk(text, chunk_size=10000):     
    c = anthropic.Client(ANTHROPIC_API_KEY)
    summaries = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        response = await c.acompletion(
            prompt=f"{anthropic.HUMAN_PROMPT}Summarize: {chunk}{anthropic.AI_PROMPT}",  
            model="claude-v1",
            max_tokens_to_sample=4000  
        ) 
        summary = response["completion"] 
        with open(f"summary_{i}.txt", "w") as f:
            f.write(summary)
        summaries.append(summary)
        time.sleep(5)
    return "\n\n".join(summaries)   

async def main(file_path):
    with open(file_path, "r") as f:
        text = f.read()
    summary = await summarize_chunk(text)
    with open("summary.txt", "w") as f:
        f.write(summary)

if __name__ == "__main__":
    file_path = sys.argv[1]
    asyncio.run(main(file_path)) 
