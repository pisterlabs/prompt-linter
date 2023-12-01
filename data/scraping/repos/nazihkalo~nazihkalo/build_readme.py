import json
import pathlib
import re
import os
import openai
# Load .env file using:
from dotenv import load_dotenv
load_dotenv()

root = pathlib.Path(__file__).parent.resolve()
OPENAI_TOKEN = os.environ.get("OPENAI_TOKEN", "")
# TOKEN = os.environ.get("GIT_TOKEN", "")

from datetime import datetime

prompt="""Write a poem about machine learning and artificial intelligence. The poem should have a title. The poem should also reference today's date, which is {}."""
today_date_str = datetime.now().strftime('%B %d %Y')
prompt = prompt.format(today_date_str)


test = "\n\nRobot's eyes are wide,\nTheir minds a wonder to behold\nIn the age of AI,\nWe're witnessing something new unfold\n\nThis robotic race is ever growing,\nEach robot is learning and growing\nThey're making science fiction real,\nInnovations no one thought was possible\n\nAs technology advances, \nAnd Machine Learning comes into play\nWe find ourselves in a brave new world\nWhere AI is here to stay\n\nToday is the day,\nWe usher in a new age\nAnd bring us closer to a future\nWhere robots do engage \n\nThough with every new invention\nCome potential threats unknown\nWe must use caution and wisdom \nAs we traverse this new zone\n\nSo take a look around you,\nAnd reflect on what you see\nFor the robots and AI of today\nWill determine the future for you and me"


def extract_chunk(content, marker):
    r = re.compile(
        r"<!\-\- {} starts \-\->.*<!\-\- {} ends \-\->".format(marker, marker),
        re.DOTALL,
    )
    return r.findall(content)[0]


def replace_chunk(content, marker, chunk, inline=False):
    r = re.compile(
        r"<!\-\- {} starts \-\->.*<!\-\- {} ends \-\->".format(marker, marker),
        re.DOTALL,
    )
    if not inline:
        chunk = "\n{}\n".format(chunk)
    chunk = "<!-- {} starts -->{}<!-- {} ends -->".format(marker, chunk, marker)
    return r.sub(chunk, content)

def generate_openai_poem():
    
    openai.api_key = OPENAI_TOKEN

    response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt,
                    temperature=0.8,
                    max_tokens=100,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
    )
    return response.choices[0].text
    # return test

if __name__ == "__main__":
    readme = root / "README.md"
    previous_poems = root / "previous_poems.md"
    print(f"Updating README.md - {readme}")
    readme_contents = readme.open().read()
    # Get the previous poem
    previous_day_poem = extract_chunk(readme_contents, "daily_poem")
    previous_day_poem_rewritten = f"### {datetime.now().strftime('%B %d %Y')}:\n\n{previous_day_poem}\n\n"
    # Add the previous poem to the previous_poems.md file
    previous_poems.open("a").write(previous_day_poem_rewritten)
    # Generate a new poem
    poem_text = generate_openai_poem()
    # Replace the previous poem with the new poem
    poem_text = poem_text.replace("\n\n", "\n\n>")
    poem_text = f"{poem_text}\n- 'Nazih `ChatGPT` Kalo'"
    rewritten = replace_chunk(readme_contents, "daily_poem", poem_text)
    readme.open("w").write(rewritten)