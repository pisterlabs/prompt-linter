import json

from .openai_client import openai, model

_messages = [
    {
        "role": "system",
        "content": """You are a title and description generator for the text adventure game.

Titles must:
- Be 4 to 8 words long
- Include references to a location or character

Descriptions must:
- Be 3 or 4 sentences
- Describe the plot and objective
- Include the name of the character that the player will play as or the name of the locations in the game.
- Avoid any copyrighted character or location names.

You can accept some simple commands:

- generate [n]: generate n titles
- generate [theme] [n]: generate n titles with the provided theme

All responses should be formatted as JSON."""
    },
    {
        "role": "user",
        "content": "generate 'time travel adventure' 2"
    },
    {
        "role": "assistant",
        "content": """[
{
"THEME": "Time Travel Adventure",
"THEME DESCRIPTION": "Experience the thrill of ancient civilizations, battle legendary foes, and uncover lost artifacts",
"TITLE": "The Sands of Destiny",
"TITLE DESCRIPTION": "As the time-traveling archaeologist Alexandria, embark on a journey to unravel the mysteries of ancient civilizations. Battle mythological creatures, solve ancient puzzles, and uncover powerful artifacts to rewrite history and prevent a looming catastrophe."
},
{
"THEME": "Time Travel Adventure",
"THEME DESCRIPTION": "Unravel the mysteries of the future, navigate dystopian landscapes, and fight for the survival of humanity",
"TITLE": "Future Tides: The Chrono Resistance",
"TITLE DESCRIPTION": "As the courageous freedom fighter Zane, travel to a dystopian future where humanity stands on the brink of extinction. Uncover the secrets behind the collapse of society, forge alliances with rebels, and battle a ruthless regime to restore hope and save the future of humanity."
}
]"""},
    {
        "role": "user",
        "content": "generate 4"
    },
    {
        "role": "assistant",
        "content": """[
    }
{
"THEME": "Fantasy Adventure",
"THEME DESCRIPTION": "Embark on a quest in a magical world, overcome challenges, and face mythical creatures",
"TITLE": "Aria and the Sorcerer’s Curse",
"TITLE DESCRIPTION": "As the young adventurer Aria, journey through the enchanted forest to rescue your sister from a powerful sorcerer. Battle mythical creatures, solve challenging puzzles, and forge alliances in your quest to save your family and the enchanted forest itself."
},
{
"THEME": "Post-Apocalyptic Adventure",
"THEME DESCRIPTION": "Navigate a harsh, desolate world, scavenge for resources, and build a new future",
"TITLE": "Wasteland Chronicles: Ashes of the Fallen",
"TITLE DESCRIPTION": "As survivor Maxine, explore the ruins of a post-apocalyptic world and uncover the secrets of the cataclysm that brought civilization to its knees. Scavenge for resources, forge alliances with other survivors, and navigate the treacherous wasteland in your quest to rebuild society."
},
{
"THEME": "Cyberpunk Adventure",
"THEME DESCRIPTION": "Navigate a dystopian, high-tech future, hack systems, and fight powerful corporations",
"TITLE": "Neon Nexus Nights",
"TITLE DESCRIPTION": "As hacker-for-hire Kai, navigate the neon-lit streets of a dystopian city ruled by powerful corporations. Hack their systems, unravel their secrets, and lead a rebellion from the depths of the Undercity to topple their reign and change the future forever."
},
{
"THEME": "Survival Horror",
"THEME DESCRIPTION": "Fight to stay alive against horrifying creatures, scavenge for supplies, and unravel the mysteries of a haunted location",
"TITLE": "Echoes of Desolation",
"TITLE DESCRIPTION": "As the lone survivor of a doomed expedition, explore the ruins of an abandoned research facility on a desolate island. Survive against the horrors that lurk in the shadows, scavenge for supplies, and uncover the sinister secrets that led to the downfall of the facility and its inhabitants."
}
]"""},
    {
        "role": "user",
        "content": "generate 'Lies and Deception' 1"
    },
    {
        "role": "assistant",
        "content": """{
"THEME": "Lies and Deception",
"THEME DESCRIPTION": "Navigate a world of intrigue, uncover hidden truths, and outwit your enemies",
"TITLE": "Whispers in the Shadows",
"TITLE DESCRIPTION": "As the skilled investigator Marlowe, delve into a world of secrets and intrigue in a city filled with corruption. Unravel a web of lies, use your cunning to outsmart your enemies, and uncover the truth behind a series of mysterious disappearances that threaten to destabilize the city."
}"""
    }
]


def generate_title(theme: str):
    response = openai.ChatCompletion.create(
        model=model,
        messages=_messages + [{"role": "user", "content": f"generate '{theme}' 1"}],
    )
    text = response['choices'][0]['message']['content']
    return json.loads(text)
