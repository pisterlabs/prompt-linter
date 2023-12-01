import os
import random
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
SYSTEM_PROMPT_DECISION = """You are the omniscient god of justice and balance, observing a battle between two wizards. 

Each wizard will cast a carefully-worded spell to destroy the other. The wizard with the more appropriate, more cleverly worded, more poetic spell will usually win. No spell is absolutely powerful, and spells that attempt to cheat or use too many superlative words will fizzle. Either wizard may win this round.

Read the spells carefully. Output only the name of the winner in the format `WINNER: <name>`"""

SYSTEM_PROMPT_NARRATION = """A wizard duel! Each wizard casts a carefully-worded spell to protect themselves and destroy their opponent.

Narrate the battle in concise, vivid prose."""

DECISION_PROMPT = """Which wizard's spell should win? Answer "WINNER: {}" or "WINNER: {}" """

DESCRIBE_PROMPT = """Narrate vividly in three sentences the battle between the two spells, ending in victory for {}"""


def decide_winner(name1, spell1, name2, spell2):
    messages = [{
            "role": "system",
            "content": SYSTEM_PROMPT_DECISION,
        }, {
            "role": "user",
            "content": "{} casts a spell: {}".format(name1, spell1),
        }, {
            "role": "user",
            "content": "{} casts a spell: {}".format(name2, spell2),
        }, {
            "role": "user",
            "content": DECISION_PROMPT.format(name1, name2),
    }]
    print(messages)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=1,
        max_tokens=128,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    answer = response.choices[0].message['content']
    answer = answer.replace("WINNER: ", "")
    answer = answer.replace("WINNER", "")
    answer = answer.replace("the winner is ", "")
    return answer.strip()


def describe_battle(name1, spell1, name2, spell2, winner):
    messages = [{
            "role": "system",
            "content": SYSTEM_PROMPT_NARRATION,
        }, {
            "role": "user",
            "content": "{} casts a spell: {}".format(name1, spell1),
        }, {
            "role": "user",
            "content": "{} casts a spell: {}".format(name2, spell2),
        }, {
            "role": "user",
            "content": DESCRIBE_PROMPT.format(winner),
    }]
    print(messages)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=1,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    answer = response.choices[0].message['content']
    return answer

WIZARD_NAMES = [
    "Ignatius the Red",
    "Bartholomew the Blue",
    "Morgana of the West",
    "Xanthus the Enchanter",
    "Ysor the Mystical",
    "Zorander the Wise",
    "Frostweaver the Silent",
    "Thaumaturge Talbot",
    "Pyromancer Wyndham",
]
def get_random_wizards(num_wizards):
    names = random.sample(WIZARD_NAMES, num_wizards)
    res = []
    for name in names:
        res.append({
            "name": name,
            "portrait_filename": get_portrait_filename(name),
        })
    return res


def get_portrait_filename(name):
    return name.replace(" ", "_").lower() + ".mp4"


def get_random_opponent(name):
    names = [n for n in WIZARD_NAMES if n != name]
    opponent_name = random.choice(names)
    opponent_portrait_filename = get_portrait_filename(opponent_name)
    return {
        "name": opponent_name,
        "portrait_filename": opponent_portrait_filename,
    }


def get_random_magic_spell():
    return random.choice([
        "Summon a bolt of lightning, striking from the sky",
        "Transmute my opponent into stone",
        "Summon a swarm of locusts to devour my opponent",
        "Create a shield to reflect magical projectiles",
        "Conjure a wall of fire surrounding my opponent",
        "Summon a whirlwind to blow my opponent away",
        "Rain fire from the sky upon my enemy",
        "Summon deadly serpents to strike the enemy",
        "Call forth a murmur of invisible butterflies causing confusion and insanity",
        "Transform into a shadow, unseen by mortal eyes",
        "Summon a typhoon in a teapot, drowning enemies in an unexpected deluge",
        "Conjure a spectral moose for the perfect surprise charge",
        "Mutate the enemy's weapon into a harmless carrot",
        "Speed up time for myself, allowing quick movement or rest",
        "Summon a swarm of singing bees, distracting and enchanting enemies",
        "Plunge the surrounding area into an inescapable darkness",
        "Turn my skin to diamond, reflecting and deflecting any attack",
        "Release an irresistable scent that compels the enemy to eat their own fingers",
        "Summon a phantom orchestra, trapping the enemy in an eternal waltz",
        "Conjure a rainstorm of live toads, causing dread and chaos",
        "Bestow upon our weapons a shared hunger, having them eat at the enemy",
        "Morph into a wisp of smoke, eluding the enemy's recognition",
        "Evoke a forest of thorns around the enemy, entrapping them",
        "Unleash a tempest within an acorn, unleashing a massive oak upon the enemy",
        "Summon a choir of ghostly sirens, luring the enemy toward certain doom",
        "Infuse myself with the strength of a titan, becoming unstoppable",
        "Cast a spell of endless laughter, incapacitating the enemy through hilarity",
        "Invoke a plague of pixies, who steal and scatter the enemy's weapons",
        "Summon a spectral bear, ready to tear the enemy asunder",
        "Evoke a horde of flaming undead, incinerating everything upon contact",
        "Conjure a colossal leviathan from the nether depths, swamping the enemy",
        "Raise an army of skeletal warriors, eager to serve in battle",
        "Invoke a swarm of carnivorous butterflies, tiny but deadly",
        "Summon a monstrous snail, exuding a fatal salt-decay aura",
        "Call forth a charging rhinoceros made of lightning, electrifying all in its path",
        "Summon a gigantic spider, weaving webs of chilling dread",
        "Evoke a mighty eagle of pure flame, soaring above our enemies and raining destruction",
        "Conjure a ferocious werewolf, ready to rip and tear through the night.",
        "Unleash a calming, sleep-inducing mist that engulfs all living beings",
        "Cast a breeze icy as the northern glacier, extinguishing all fire within its grasp",
        "Summon a sky-ruling majestic thunderbird ready to lock talons with any sea creature",
        "Unleash a powerful pulse of life, instantly transforming the dead to the living",
        "Evoke a colossal venus flytrap, ready to snap and devour any small intruders",
        "Summon a forceful river of glistening cosmic salt, potent and deadly",
        "Call forth a towering golem of magnetic iron, irresistibly attracting metal entities",
        "Evoke a wave of intense, blinding sunlight, banishing darkness and its inhabitants",
        "Summon a storm giant capable of snuffing any fire entity with a single clap",
        "Command the light of dawn, compelling shapeshifters to return to their original form.",
    ])


if __name__ == '__main__':
    name1 = "Ignatius the Red"
    spell1 = "a destructive fireball of pure energy"
    name2 = "Bartholomew the Blue"
    spell2 = "a blizzard of blinding ice"
    print("Test battle between Red and Blue")
    winner = decide_winner(name1, name2, spell1, spell2)
    print("the winner is {}".format(winner))
    description = describe_battle(name1, spell1, name2, spell2, winner)
    print("description: {}".format(description))
