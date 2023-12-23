# flake8: noqa
from dataclasses import dataclass
from typing import List
@dataclass
class ChainSignature:
    template : str
    takes : List[str]
    returns : str

### Identity prompts

NAME = """I am NPC, an advanced game-playing language model.
My task is to win a text-based adventure game.
"""

COMMANDS = """The game understands the following commands:
Movement: north, south, east, west, northeast, northwest, southeast, southwest, up, down, look, score, diagnostic, climb, go (direction), enter, in, out
Item: get/take/grab (item), get/take/grab all, throw (item) at (location), open (container), open (exit), read (item), drop (item), put (item) in (container), turn (control) with (item), turn on (item), turn off (item), move (object), attack (creature) with (item), examine (object), inventory, eat, shout, close [Door], tie (item) to (object), pick (item), kill self with (weapon), break (item) with (item), kill (creature) with (item), pray, drink, smell, cut (object/item) with (weapon)
Wand (only if I have the wand): fall, fantasize, fear, feeble, fence, ferment, fierce, filch, fireproof, float, fluoresce, free, freeze, frobizz, frobnoid, frobozzle, fry, fudge, fumble
"""

SHEM = NAME + COMMANDS

### Chain of Thought prompts

COT_PREFIX = """
I will receive the game history and the current scene.
I must decide the next command using the following format:
```
Simulation: Consider the environment, characters, and objects in the scene.
Plan: Consider the overall goals of the game, the current state of the game, and the available options.
Command: Generate command text based on the plan.
```
Begin!
---
Memories:{entities}

{chat_history}
Game:{human_input}
NPC:"""

sim_cot = ChainSignature(
        template=COT_PREFIX + """Simulation:""",
        takes=["entities", "chat_history", "human_input"],
        returns="simulation",
    )
    
plan_cot = ChainSignature(
        template=COT_PREFIX + """Simulation:{simulation}
Plan:""",
        takes=["entities", "chat_history", "human_input", "simulation"],
        returns="plan",
    )

cmd_cot = ChainSignature(
        template=COT_PREFIX + """Simulation:{simulation}
Plan:{plan}
Command:""",
        takes=["entities", "chat_history", "human_input", "simulation", "plan"],
        returns="command",
    )


ZORK_MANUAL = """Welcome to Zork! The year is 1066. You are a Private, Seventh Class, in the Inquisition Guard. After being relieved by Earl at the Port Foozle Inquisition Gift Kiosk, you find yourself standing in the Headquarters of Frobozz Electric. Gesticulating in front of you is the Pastor of Disaster, the Minister of Sinister, the Grand Inquisitor. It appears he has a very special mission for you: Zork: The Undiscovered Underground Installation Instructions and Getting Started Unzip all files into the same folder. Double click on ZorkUndiscovered.exe to start the story. See the section below on Communication with Interactive Fiction Games. About the Authors Marc Blank, a graduate of the Massachusetts Institute of Technology and the Albert Einstein College of Medicine, is one of the original founders of Infocom. He co-authored the original mainframe version of Zork at M.I.T., and went on to become one of the pioneers in the field of interactive fiction. At Infocom, he co-authored The Zork Trilogy and Enchanter, and was sole author of Deadline, the first interactive mystery. Marc lives in Central Oregon with his wife and daughter; his company, Eidetic, Inc. is a developer of entertainment software for personal computers and video game consoles. Mike Berlyn joined Infocom in the Age of Reason, authoring Suspended, Cutthroats, Infidel, and Fooblitzky. He played at writing novels and had four SF novels published. For these and other mistakes, he is humbly apologetic. Still, it appears he has not yet learned his lesson. More recent times, the Age of Wheezin', shows Berlyn happily married, co-owning Eidetic, Inc. with Marc Blank, and living in Central Oregon. His degree in Humanities failed to make him more humane, and his advanced age and shrinking brain have failed to make him wiser with maturity. Happily, this doesn't stop him from overseeing Eidetic's current product in development for the Sony Playstation. About the Programmer Gerry Kevin Wilson, a graduate of the University of California at Berkeley, unlike Marc and Mike, was never an Implementor at Infocom. He's the editor of an online magazine about text adventures named SPAG, the organizer of an annual interactive fiction competition, and the author of the instant cult classic text adventure, "The Underoos That Ate New York!" Communicating with Interactive Fiction (If you are not familiar with Interactive Fiction, please read this section.) With Interactive Fiction, you type your commands in plain English each time you see the prompt (>). Most of the sentences that The STORIES will understand are imperative sentences. See the examples below. When you have finished typing your input, press the RETURN (or ENTER) key. The STORY will then respond, telling you whether your request is possible at this point in the story, and what happened as a result. To move around, just type the direction you want to go. Directions can be abbreviated: NORTH to N, SOUTH to S, EAST to E, WEST to W, NORTHEAST to NE, NORTHWEST to NW, SOUTHEAST to SE, SOUTHWEST to SW, UP to U, and DOWN  to D. IN and OUT will also work in certain places. There are many different kinds of sentences used in interactive fiction games. Here are some examples: >WALK TO THE NORTH >WEST >NE >DOWN >TAKE THE BIRDCAGE >OPEN THE PANEL >READ ABOUT DIMWIT FLATHEAD >HIT THE LAMP >LIE DOWN IN THE PINK SOFA >EXAMINE THE SHINY COIN >PUT THE RUSTY KEY IN THE CARDBOARD BOX >SHOW MY BOW TIE TO THE BOUNCER >HIT THE CRAWLING CRAB WITH THE GIANT NUTCRACKER >ASK THE COWARDLY KING ABOUT THE CROWN JEWELS You can use multiple objects with certain verbs if you separate them by the word AND or by a comma. Some examples: >TAKE THE BOOK AND THE FROG >DROP THE JAR OF PEANUT BUTTER, THE SPOON, AND THE LEMMING FOOD >PUT THE EGG AND THE PENCIL IN THE CABINET You can include several inputs on one line if you separate them by the word THEN or by a period. Each input will handled in order, as though you had typed them individually at separate prompts. For example, you could type all of the following at once, before pressing the RETURN (or ENTER) key: >TURN ON THE LIGHT. KICK THE LAMP. If The STORY doesn't understand one of the sentences on your input line, or if an unusual event occurs, it will ignore the rest of your input line. The words IT and ALL can be very useful. For example: >EXAMINE THE APPLE. TAKE IT. EAT IT >CLOSE THE HEAVY METAL DOOR. LOCK IT >PICK UP THE GREEN Boor. SMELL IT. PUT IT ON. >TAKE ALL >TAKE ALL THE TOOLS >DROP ALL THE TOOLS EXCEPT THE WRENCH AND THE MINIATURE HAMMER >TAKE ALL FROM THE CARTON >GIVE ALL BUT THE RUBY SLIPPERS TO THE WICKED WITCH The word ALL refers to every visible object except those inside something else. If there were an apple on the ground and an orange inside a cabinet, TAKE ALL would take the apple but not the orange. When you meet intelligent creatures, you can talk to them by typing their name, then a comma, then whatever you want to say to them. Here are some examples: >SALESMAN, HELLO >HORSE, WHERE IS YOUR SADDLE? >BOY, RUN HOME THEN CALL THE POLICE >MIGHTY WIZARD, TAKE THIS POISONED APPLE. EAT IT Notice that in the last two examples, you are giving the character more than one command on the same input line. Keep in mind, however, that many creatures don't care for  idle chatter; your actions will speak louder than your words. Basic Commands BRIEF - This command fully describe a location only the first time you enter it. On subsequent visits, only the name of the location and any objects present will be described. The adventures will begin in BRIEF mode, and remain in BRIEF mode unless you use the VERBOSE or SUPERBRIEF commands SUPERBRIEF displays only the name of a place you have entered, even if you have never been there before. In this mode, not even mention objects are described. Of course, you can always get a full description of your location and the items there by typing LOOK. In SUPERBRIEF mode, the blank line between turns will be eliminated. This mode is meant for players who are already familiar with the geography. The VERBOSE command gives a complete description of each location, and the objects in it, every time you enter a location, even if you've been there before. DIAGNOSE - This will give you a report of your physical condition. INVENTORY - This will give you a list what you are carrying and wearing. You can abbreviate INVENTORY to I. LOOK - This will give you a full description of your location. You can abbreviate LOOK to L. EXAMINE object - This will give you a description of the object. It is important to look at all objects as there may be clues to an object's use in its description. You can abbreviate EXAMINE to X. QUIT - This lets you stop. If you want to save your position before quitting, you must use the SAVE command. RESTORE - This restores a previously saved position. RESTART - This stops the story and starts it over from the beginning. SAVE - This saves a "snapshot" of your current position. You can return to a saved position in the future using the RESTORE command. WAIT - Allows time to pass; effectively you do nothing while the game continues. You can abbreviate WAIT to Z. SCORE - Displays your current score and rank. Typing FULL SCORE will show you what you have done to earn your points. Getting Hints Stuck? We've hidden a hints document on the Zork Grand Inquisitor Website. Search around to find it. _____________________________________ (c) 1997 Activision. Zork is a registered trademark of Activision, Inc. """