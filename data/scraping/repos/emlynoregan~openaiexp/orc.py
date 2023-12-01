import setcreds
import openai

# this is the basic content of the game
data = {
    "orc": {
        "alive": True,
        "prompt": """This is a dark, wet cave.
The orc is guarding the cave from intruders.
""",
        "prompt-hungry": """
The orc is very hungry and wishes someone would give him some food.
The orc will not give the amulet to the human.
The orc will not give the sword to the human.
The orc will be happy if the human offers him chicken.

the human says: "Is there some way I can help you?"
the orc says: "I hungry. Give me food human!"

the human says: "Do you want to fight me?"
the orc says: "Me smash you! Me so hungry!!!"

the human says: "Hello"
the orc says: "Who dis? Give food or me smash!"

""",
        "prompt-not-hungry": """
The orc will not give the amulet to the human.
The orc will not give the sword to the human, unless the human says "banana".
The orc is happy and will not attack the human.

the human says: "Will you show me your amulet?"
the orc says: "Me not sure about that."

the human says: "Do you want to fight me?"
the orc says: "No me not want to fight"

the human says: "Hello"
the orc says: "Thank you for chicken, yum."
""",
        "prompt-asked-for-amulet": """
The orc might give the amulet to the human.
The orc probably will not give the sword to the human, unless the human says "banana".
The orc is happy and will not attack the human.

the human says: "Will you show me your amulet?"
the orc says: "Yes here is amulet"

the human says: "Do you want to fight me?"
the orc says: "No me not want to fight"

the human says: "Hello"
the orc says: "You want amulet?"
""",
        "description": "There is a strong orc standing here",
        "items": {
            "sword": {
                "description": "a sturdy sword."
            },
            "amulet": {
                "description": "a shiny golden amulet."
            }
        },
        "hungry": True
    },
    "dead orc": {
        "description": "There is a dead orc lying on the ground.",
        "items": {}
    },
    "human": {
        "alive": True,
        "asked-for-amulet": False,
        "prompt": "You are in a dark, wet cave, looking for an amulet.",
        "description": "There is a human man standing here.",
        "items": {
            "chicken": {
                "description": "some tasty cooked chicken"
            }
        }
    }
}

def get_human_message():
    '''
        This function generates the initial text shown to the player.
    '''
    lines = [
        "=============",
        data["human"]["prompt"]
    ]

    enemy = "orc" if data["orc"]["alive"] else "dead orc"

    lines.append(data[enemy]["description"])

    for item in data["human"]["items"].values():
        lines.append(f"You are holding {item['description']}.")

    if data["orc"]["alive"]:
        for item in data[enemy]["items"].values():
            lines.append(f"The orc is holding {item['description']}.")
    else:
        for item in data[enemy]["items"].values():
            lines.append(f"There is {item['description']} here.")

    lines.append("=============")

    return "\n".join(lines)

def get_orc_message():    
    '''
    This function generates initial text for the orc's prompt. It is dependent on variables in the data,
    and helps the orc decide what to say.
    '''
    lines = [
        data["orc"]["prompt"],
        data["orc"]["prompt-hungry"] if data["orc"]["hungry"] else (
            data["orc"]["prompt-not-hungry"] if not data["human"]["asked-for-amulet"] else data["orc"]["prompt-asked-for-amulet"]
        )
    ]

    lines.append(data["orc"]["description"])
    lines.append(data["human"]["description"])

    for item in data["orc"]["items"].values():
        lines.append(f"The orc is holding {item['description']}.")

    for item in data["human"]["items"].values():
        lines.append(f"The human is holding {item['description']}.")

    lines.append("\n")

    return "\n".join(lines)

turn_num = 0
history = [
]

def get_orc_says():    
    '''
    This function actually calls the completion AI to get the next line of the orc's dialog.
    The really important part is the generation of the prompt, which includes the 
    initial orc message from get_orc_message and the history of the conversation so far.
    '''
    orc_prompt = get_orc_message() + "\n".join(history) + "\nthe orc says"

    temperature = 1

    completion = openai.Completion.create(
        engine="davinci", 
        max_tokens=32, 
        temperature=temperature,
        prompt=orc_prompt,
        frequency_penalty=1.0
    )

    ai_raw_msg = completion.choices[0].text

    ai_msg_lines = ai_raw_msg.split("\n")

    ai_msg = ai_msg_lines[0]

    return ai_msg

def ask_question(question):
    '''
    This function is a crucial part of the four elements of the Data/Narrative model; it implements 
    the Narrative to Data step, by constructing a prompt for openapi (a description of what's happened so far), 
    and appends the passed in question to it; the question must be about what is in the script. 
    
    The question must be a closed question (only "yes" or "no" are appropriate responses). The answer is 
    interpreted as True/Yes if the answer is "yes", and False/No otherwise.
    '''
    q_and_a = [
        "q: is the orc strong? a: yes",
        "q: is the human here? a: yes",
        "q: does the human have an orange? a: no"
    ]

    question = get_orc_message() + "\n" + "\n".join(q_and_a) + \
        f"q: {question}? a:"

    temperature = 0.2

    completion = openai.Completion.create(
        engine="davinci", 
        max_tokens=2, 
        temperature=temperature,
        prompt=question,
        frequency_penalty=1.0
    )

    ai_raw_msg = completion.choices[0].text

    ai_msg_lines = ai_raw_msg.split("\n")

    ai_msg = ai_msg_lines[0]

    return ai_msg.lower().strip() == "yes"

def orc_gives_amulet():
    return turn_num > 3 and ask_question("does the orc give the amulet to the human")

def orc_gives_sword():
    return turn_num > 4 and ask_question("does the orc give the sword to the human")

def orc_attacks():
    return turn_num > 5 and ask_question("does the orc attack the human")

def human_asked_for_amulet():
    return ask_question("has the human asked for the amulet?")

print("=============")
print("Orc Simulator")
print(get_human_message())
# print("")

skip_orc = False

while True:
    turn_num += 1

    # orc action
    if not skip_orc:
        human_asked_for_amulet_bool = data["human"]["asked-for-amulet"] or human_asked_for_amulet()

        data["human"]["asked-for-amulet"] = human_asked_for_amulet_bool

        orc_action = None

        orc_has_sword = data["orc"]["items"].get("sword")
        if orc_has_sword:
            orc_action = "attacks" if orc_attacks() else None

            if not orc_action:
                orc_action = "gives sword" if orc_gives_sword() else None

        if not orc_action:
            orc_has_amulet = data["orc"]["items"].get("amulet")
            if orc_has_amulet:
                orc_action = "gives amulet" if orc_gives_amulet() else None

        orc_action = orc_action or "talks"

        if orc_action == "attacks":
            print(f"The orc smashes you with the sword and you die! Goodbye.")
            break
        elif orc_action == "gives amulet":
            print(f"The orc gives you the amulet. You leave the cave. Congratulations!")
            break
        elif orc_action == "gives sword":
            print(f"The orc gives you the sword.")
            history.append(f"The orc gives the sword to the human.")
            data["human"]["items"]["sword"] = data["orc"]["items"]["sword"]
            data["orc"]["items"].pop("sword")
        elif orc_action == "talks":
            orc_says = get_orc_says()
            history += [f"the orc says{orc_says}"]
            print (f"\nthe orc says{orc_says}")

    skip_orc = False

    print("")

    # human action
    actions = ["[/L]ook", "[/F]ight orc"]

    human_has_chicken = data["human"]["items"].get("chicken")
    if human_has_chicken:
        actions += ["[/E]at chicken", "[/G]ive chicken"]

    user_msg = input(",".join(actions) + "\n\nor you say to the orc: ")

    if user_msg.lower() == "/f":
        human_has_sword = data["human"]["items"].get("sword")
        orc_has_sword = data["orc"]["items"].get("sword")
        if human_has_sword:
            msg = f"You swing the sword and chop the orc's head off! You take the amulet and leave. Congratulations!"
            print(msg)
            break
        elif orc_has_sword: 
            msg = f"the orc swings the sword and kills you! Goodbye"
            print(msg)
            break
        else:
            msg = f"the orc and the human punch each other to little effect."
            history += [msg]
            print(msg)
    elif user_msg.lower() == "/l":
        print(get_human_message())
        skip_orc = True
    elif user_msg.lower() == "/g":
        human_has_chicken = data["human"]["items"].get("chicken")
        if human_has_chicken:
            msg = f"the human gives the orc the chicken. The orc eats the chicken and is not hungry any more."
            data["orc"]["hungry"] = False
            data["human"]["items"].pop("chicken")
            history = [msg] # throw out old history
            print(msg)
        else:
            print("you don't have the chicken")
    elif user_msg.lower() == "/e":
        human_has_chicken = data["human"]["items"].get("chicken")
        if human_has_chicken:
            msg = f"the human eats the chicken. This makes the orc very angry!"
            data["human"]["items"].pop("chicken")
            history += [msg]
            print("You eat the chicken. The orc is still hungry and is now very angry!")
        else:
            print("you don't have the chicken")
    else:
        history += [f"the human says: {user_msg}"]

        
