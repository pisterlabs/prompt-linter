import openai
import random
from utilities import stream_print

openai.api_key = "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"
#engine_type = "text-ada-001"
engine_type = "text-davinci-003"

class Joust:

    def __init__(self, combatant_A_name="", combatant_B_name=""):
        self.combatant_A_name = combatant_A_name
        self.combatant_B_name = combatant_B_name

    def main_joust(self):
        
        if self.combatant_A_name == "":
            self.combatant_A_name = self.generate_knight_name();
        if self.combatant_B_name == "":
            self.combatant_B_name = self.generate_knight_name();

        prompt = self.setup_joust(self.combatant_A_name, self.combatant_B_name)
        print(f"We are gathered here to watch a joust between two brave " + \
              f"knights: {self.combatant_A_name}, and {self.combatant_B_name}.")
        print("Who will win on this perilous day? We shall soon find out!\n")

        
        while(True):
            user_input = self.get_joust_input(self.combatant_A_name, \
                                              self.combatant_B_name)

            if (user_input[0] == 0):
                winner_name = self.combatant_B_name
            elif (user_input[0] == 1):
                winner_name = self.combatant_A_name
            else:
                continue

            prompt += self.designate_joust_winner(winner_name, user_input)
            self.generate_joust(prompt)
            print("")
            input()
            break


    def designate_joust_winner(self, winner, details):
        prompt = (winner + " will win the joust by aiming " + details[1].lower() \
                  + " because the opponent aimed " + details[2].lower() + ".")
        return prompt


    #this function returns a 1 if the player won, and a 0 if the computer won
    def get_joust_input(self, combatant_A_name="user", combatant_B_name="comp"):
        possible_actions = ["high", "straight", "low"]
        print("Do you aim high, straight, or low?")
        user_action = input().lower()
        if user_action:
            if user_action == "h" or "s" or "l":
                if user_action == "h": user_action = "high"
                if user_action == "m": user_action = "straight"
                if user_action == "l": user_action = "low"
            print(f"{combatant_A_name} aims {user_action}.")
        else:
            print(f"{combatant_A_name} can't decide, and picks randomly.")
            user_action = random.choice(possible_actions)
            print(f"The winds of fate decrees their lance will go {user_action}!")
        possible_actions.remove(user_action)
        computer_action = random.choice(possible_actions)
        
        if user_action:
            pass
        else:
            user_action = random.choice(possible_actions).capitalize()
        
        if user_action == "high":
            if computer_action == "straight":
                result = 1
            elif computer_action == "low":
                result = 0
        elif user_action == "straight":
            if computer_action == "low":
                result = 1
            elif computer_action == "high":
                result = 0
        elif user_action == "low":
            if computer_action == "high":
                result = 1
            elif computer_action == "straight":
                result = 0
        else:
            print("Could not parse input. Sorry.")
        return (result, user_action, computer_action)


    def setup_joust(self, A, B):
        
        if A and B:
            prompt = "Describe a joust between two knights named " + A + \
                     " and " + B + ".\n\n"
        else:
            prompt = "Describe a joust between two nameless knights."
        return prompt


    def generate_joust(self, prompt):
        response = openai.Completion.create(engine=engine_type, \
                                            prompt=prompt, \
                                            max_tokens = 512, \
                                            temperature = 1, \
                                            stream=True)
        collected_events = []
        completion_text = ''
        self.character_count = 0

        for event in response:
            collected_events.append(event)
            event_text = event['choices'][0]['text']
            completion_text += event_text
            self.character_count = stream_print(event_text, \
                                                self.character_count) 

    def generate_knight_name(self):
        prompt = "Write the name of a knight who is partaking in a joust.\n\n" + \
                 "Here are four good examples:\n\n" + \
                 "Sir Arthilon, the blue and gold knight of Papilae\n" + \
                 "Ser Bureaugard, the Brave\n" + \
                 "Oxelot of the Golden Forest\n"
        response = openai.Completion.create(engine=engine_type, \
                                            prompt=prompt, \
                                            max_tokens = 16, \
                                            temperature = 1)
        return response.choices[0].text.strip()
