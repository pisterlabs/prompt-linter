import pickle
from utils import OpenAIAsync
import asyncio
import dotenv
import os
import time

dotenv.load_dotenv()

if __name__ == "__main__":
    with open("data/feature_engineering/metar_strings.pickle", "rb") as pickle_file:
        metar_strings = pickle.load(pickle_file)

    openai_instance = OpenAIAsync(os.getenv("OPENAI_API_KEY"))

    step = 500
    for i in range(0, len(metar_strings), step):
        time.sleep(90)
        print(f"Starting batch {i} to {i+step}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        loop.run_until_complete(
            openai_instance.main(
                metar_strings[i : i + step],
                output_directory="data/metar_scores_llm",
                file_name=f"metar_results_{i}.json",
            )
        )
