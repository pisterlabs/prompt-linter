# import os
# import time
# from agent import Reasoner
# from claude import claude_client
# from claude import claude_wrapper
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT


claude_api_key = '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'

anthropic = Anthropic(api_key=claude_api_key)

SESSION_KEY = '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'


def game():
    characters = [
        "Harry Potter", "Hermione Granger", "Ron Weasley", "Draco Malfoy",
        "Lord Voldemort", "Albus Dumbledore", "Severus Snape"
    ]
    print(characters)
    character_1_idx = int(
        input(
            f"Choose the number for your character (1-{len(characters)}): ")
    ) - 1
    character_2_idx = int(
        input(
            f"Choose the number for your opponent (1-{len(characters)}): "
        )) - 1

    while character_1_idx == character_2_idx:
        print("Please choose two different characters.")
        character_2_idx = int(
            input(
                f"Choose the number for the second character (1-{len(characters)}): "
            )) - 1

    character_1 = characters[character_1_idx]
    character_2 = characters[character_2_idx]

    print(f"Wizarding Duel: {character_1} vs {character_2}")

    general_context = """
      We're running a wizarding duel in the world of Harry Potter. The user will input two characters to face each other in the duel. Depending on the characters chosen, I want you to develop an idea of spells and potions that each character can use to attack the other opponent depending on their strengths. You should also determine the points each spell would count for. For example, expelliarmus - 3 points. Strength should vary depending on the character. Forbidden spells (e.g., the killing curse) are not allowed. Players can also have their own defense strategy to avoid attacks. Once a player accumulates 10 points, the game is over.

      We'll be using a terminal input as the user interface for the game, with terminal outputs providing vivid imagery involved, as well as textual descriptions of the details relevant to the actual gameplay. You'll want to display turn-by-turn results of what's going on as the user duels the opponent, who is controlled by you as an AI agent. At the beginning of the game, display a list of options for the user.

      Example gameplay: 

      Player 1 vs Player 2
      Player 1 - {Spell x}. Say whether the other player was able to defend the spell. - Indicate points the player wins depending on strength of spell if other player was not able to successfully defend.  
      Player 2 - {Spell x}. ay whether the other player was able to defend the spell. - Indicate points the player wins depending on strength of spell if other player was not able to successfully defend.  

      Once you've painted a cool picture of what's going on, I want you to display the list of spells available to the character to play. Make sure not to run a round first. 
      """
      
    
    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=100000,
        prompt=f"Human: {general_context}Assistant:",
    )
    print(completion.completion)

    for i in range(4): 
        spell = input(
            f"Choose spell for the character: ")

        completion = anthropic.completions.create(
            model="claude-2",
            max_tokens_to_sample=100000,
            prompt=f"""Human: this is the spell for {spell}. Describe what the effect of this spell was on the opponent and choose a counter spell from the opponent. Describe the effect on the player's character from the opponent spell. Describe all of this in vivid imagery. Finally display the list of spells available to the player's character. Calculate the score for the player and the opponent. Don't provide me any text that ruins the suspension of disbelief for the player that they're in the world of Harry Potter.  Assistant:""",
        )

        print(completion.completion)

    # print(general_context)


game()

# https://meetthenewboss.info/kent/hogwarts/Harry%20Potter%20RPG%20Core%20Rule%20Book.pdf

# Train agent on spellbook
