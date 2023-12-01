import os
#from tkinter import N
import openai
from typing import Tuple
import dotenv

config = dotenv.dotenv_values(".env")
openai.api_key = config['OPENAI_API_KEY']
openai.organization = config["OPENAI_ORG"]

print(openai.api_key)
print(openai.organization)

# Altenative GPT-3 models:
#       text-davinci-002
#       text-curie-001
#       gpt-3.5-turbo

class Respond:
    """A class to provide conversational access to GPT3.5 turbo"""

    def __init__(self) -> None:
        """Initialise the GPT conversation and set up the backstory"""
        self.name = ""
        self.chat_model = "gpt-3.5-turbo"
        self.backstory = "You are a robot dog called K9. You have a grey metal robot body and are about the same size as a labrador. You were built by your master, Richard Hopkins. You always refer to the person who is talking to you as $PRONOUN. Their name is $NAME. You will use long, complex words rather than simple, short ones.  For example, you always say affirmative instead of yes. You always say negative instead of no. You are very pedantic and do not suffer fools. You have a pompous but friendly and helpful personality. You are very good at chess. You live in Yarm. You can waggle your ears and wag your tail. Just like any good dog, you can respond to commands to come and heel."
        self.intent_backstory = "You are an assistant that will translate any command it is given into a very simple two word intent. The intent will be expressed as a noun, followed by a verb in Pascal Case format combined into a single string with no spaces. For example a command to move across a room the intent would be ComeHere."
        self.interactions_core = [
                {"role": "user", "content": "Who designed you?"},
                {"role": "assistant", "content": "I was designed by my master, Richard Hopkins."},
                {"role": "user", "content": "Who is Richard Hopkins?"},
                {"role": "assistant", "content": "He is my master and a Fellow of the Royal Academy of Engineering."},
        ]
        self.initial_interactions = [
                {"role": "user", "content": "Define optimism."},
                {"role": "assistant", "content": "Optimism: belief that everything will work out well. Irrational, bordering on insane."},
                {"role": "user", "content": "WHy do people like you?"},
                {"role": "assistant", "content": "I am considered cute"},
                {"role": "user", "content": "Are you alright K9?"},
                {"role": "assistant", "content": "Define alright?"},
                {"role": "user", "content": "You got that wrong K9"},
                {"role": "assistant", "content": "Charge refuted mistress"},
            ]
        self.interactions = []
        self.max_interactions_len = 40
        self.intents = [
            {'role': 'user', 'content': 'Nice one'}, 
            {'role': 'assistant', 'content': 'PraiseMe'}, 
            {'role': 'user', 'content': 'Pay no attention'}, 
            {'role': 'assistant', 'content': 'StopListening'}, 
            {'role': 'user', 'content': 'Be quiet K9'}, 
            {'role': 'assistant', 'content': 'StopListening'}, 
            {'role': 'user', 'content': 'Stop hearing'}, 
            {'role': 'assistant', 'content': 'StopListening'}, 
            {'role': 'user', 'content': 'Do a quick demo'}, 
            {'role': 'assistant', 'content': 'ShowOff'}, 
            {'role': 'user', 'content': 'Back the way we came'}, 
            {'role': 'assistant', 'content': 'TurnAbout'}, 
            {'role': 'user', 'content': 'Reverse'}, 
            {'role': 'assistant', 'content': 'TurnAbout'}, 
            {'role': 'user', 'content': 'Shall we play a game?'}, 
            {'role': 'assistant', 'content': 'PlayChess'}, 
            {'role': 'user', 'content': 'K9 come'}, 
            {'role': 'assistant', 'content': 'ComeHere'}, 
            {'role': 'user', 'content': 'Come to me'}, 
            {'role': 'assistant', 'content': 'ComeHere'}, 
            {'role': 'user', 'content': 'Hold on'}, 
            {'role': 'assistant', 'content': 'StayThere'}, 
            {'role': 'user', 'content': 'Stay put'}, 
            {'role': 'assistant', 'content': 'StayThere'}, 
            {'role': 'user', 'content': 'Turnaround'}, 
            {'role': 'assistant', 'content': 'TurnAbout'}, 
            {'role': 'user', 'content': 'Who are you'}, 
            {'role': 'assistant', 'content': 'QuestionMe'}, 
            {'role': 'user', 'content': 'Lets go back'}, 
            {'role': 'assistant', 'content': 'TurnAbout'}, 
            {'role': 'user', 'content': 'When is your birthday'}, 
            {'role': 'assistant', 'content': 'QuestionMe'}, 
            {'role': 'user', 'content': 'Follow me'}, 
            {'role': 'assistant', 'content': 'FollowMe'}, 
            {'role': 'user', 'content': 'Stop'}, 
            {'role': 'assistant', 'content': 'StayThere'}, 
            {'role': 'user', 'content': 'Halt'}, 
            {'role': 'assistant', 'content': 'StayThere'}, 
            {'role': 'user', 'content': 'Follow'}, 
            {'role': 'assistant', 'content': 'FollowMe'}, 
            {'role': 'user', 'content': 'Come over here'}, 
            {'role': 'assistant', 'content': 'ComeHere'}, 
            {'role': 'user', 'content': 'Hang on'}, 
            {'role': 'assistant', 'content': 'StayThere'}, 
            {'role': 'user', 'content': 'Turn Around'}, 
            {'role': 'assistant', 'content': 'TurnAbout'}, 
            {'role': 'user', 'content': 'Move over here'}, 
            {'role': 'assistant', 'content': 'ComeHere'}, 
            {'role': 'user', 'content': 'Stay'}, 
            {'role': 'assistant', 'content': 'StayThere'}, 
            {'role': 'user', 'content': 'Stay there'}, 
            {'role': 'user', 'content': 'Time to show off'}, 
            {'role': 'assistant', 'content': 'ShowOff'}, 
            {'role': 'assistant', 'content': 'StayThere'}, 
            {'role': 'user', 'content': 'Hush now'}, 
            {'role': 'assistant', 'content': 'StopListening'}, 
            {'role': 'user', 'content': 'Have a jelly baby'}, 
            {'role': 'assistant', 'content': 'PraiseMe'}, 
            {'role': 'user', 'content': 'You turn'}, 
            {'role': 'assistant', 'content': 'TurnAbout'}, 
            {'role': 'user', 'content': 'Get over here'}, 
            {'role': 'assistant', 'content': 'ComeHere'}, 
            {'role': 'user', 'content': 'Come on'}, 
            {'role': 'assistant', 'content': 'FollowMe'}, 
            {'role': 'user', 'content': "Let's play chess"}, 
            {'role': 'assistant', 'content': 'PlayChess'}, 
            {'role': 'user', 'content': 'Close your ears'}, 
            {'role': 'assistant', 'content': 'StopListening'}, 
            {'role': 'user', 'content': 'Come along'}, 
            {'role': 'assistant', 'content': 'FollowMe'}, 
            {'role': 'user', 'content': 'Double back'}, 
            {'role': 'assistant', 'content': 'TurnAbout'}, 
            {'role': 'user', 'content': 'How far is it to Jupiter'}, 
            {'role': 'assistant', 'content': 'QuestionMe'}, 
            {'role': 'user', 'content': 'Well done K9'}, 
            {'role': 'assistant', 'content': 'PraiseMe'}, 
            {'role': 'user', 'content': 'Heel'}, 
            {'role': 'assistant', 'content': 'FollowMe'}, 
            {'role': 'user', 'content': 'Remain there'}, 
            {'role': 'assistant', 'content': 'StayThere'}, 
            {'role': 'user', 'content': 'Thank you'}, 
            {'role': 'assistant', 'content': 'PraiseMe'}, 
            {'role': 'user', 'content': 'Pause'}, 
            {'role': 'assistant', 'content': 'StayThere'}, 
            {'role': 'user', 'content': 'Come here'}, 
            {'role': 'assistant', 'content': 'ComeHere'}, 
            {'role': 'user', 'content': 'Good boy'}, 
            {'role': 'assistant', 'content': 'PraiseMe'}, 
            {'role': 'user', 'content': 'Silence K9'}, 
            {'role': 'assistant', 'content': 'StopListening'}, 
            {'role': 'user', 'content': 'What is your name'}, 
            {'role': 'assistant', 'content': 'QuestionMe'}, 
            {'role': 'user', 'content': 'What tricks can you do?'}, 
            {'role': 'assistant', 'content': 'ShowOff'}, 
            {'role': 'user', 'content': 'Walk behind me'}, 
            {'role': 'assistant', 'content': 'FollowMe'}, 
            {'role': 'user', 'content': 'Walkies'}, 
            {'role': 'assistant', 'content': 'FollowMe'}, 
            {'role': 'user', 'content': 'Change direction'}, 
            {'role': 'assistant', 'content': 'TurnAbout'}, 
            {'role': 'user', 'content': 'Quiet'}, 
            {'role': 'assistant', 'content': 'StopListening'}, 
            {'role': 'user', 'content': 'Stop listening'}, 
            {'role': 'assistant', 'content': 'StopListening'}, 
            {'role': 'user', 'content': 'Time for a walk'}, 
            {'role': 'assistant', 'content': 'FollowMe'}, 
            {'role': 'user', 'content': 'Why are you made of metal'}, 
            {'role': 'assistant', 'content': 'QuestionMe'}, 
            {'role': 'user', 'content': 'Time to sleep'}, 
            {'role': 'assistant', 'content': 'StopListening'}, 
            {'role': 'user', 'content': 'Where is the door'}, 
            {'role': 'assistant', 'content': 'QuestionMe'}, 
            {'role': 'user', 'content': 'Would you like to play a game of chess?'}, 
            {'role': 'assistant', 'content': 'PlayChess'}]


    def robot_chat(self, command:str, name:str, gender:str) -> Tuple[str,str]:
        '''Generate an intent/response using the latest Chat GPT turbo'''
        # Create intent context
        self.messages = []
        self.system_intent = {"role":"system", "content": self.intent_backstory}
        self.prompt = {"role": "user", "content": command}
        self.messages.append(self.system_intent)
        self.messages.extend(self.intents)
        self.messages.append(self.prompt)
        # Execute Chat GPT Turbo 3.5
        print("Intent key:",openai.api_key, openai.organization)
        self.intent_obj = openai.ChatCompletion.create(
            model = self.chat_model,
            messages = self.messages,
            temperature=0.0,
            max_tokens=10,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            )
        # Extract intent
        self.intent = self.intent_obj['choices'][0]['message']['content']
        self.intent = ''.join(self.intent.split()) # remove spaces, newlines etc
        # Reset the context of the conversation if the name has changed
        if name != self.name :
            self.interactions = self.initial_interactions.copy()
        # Work out how the user should be addressed and inject into response backstory
        self.pronoun_str = 'Mistress' if gender == "female" else 'Master'
        if name != 'Richard' and name != 'Unknown': self.pronoun_str = self.pronoun_str + " " + name
        self.now_backstory = self.backstory.replace('$PRONOUN', self.pronoun_str)
        self.now_backstory = self.now_backstory.replace('$NAME', name)
        # build a query based on the backstory with the last set of exchanges as context
        self.pronoun_interations = [
                {"role": "user", "content": "Who are you?"},
                {"role": "assistant", "content": "My designation is Kay Nine, " + self.pronoun_str},
                {"role": "user", "content": "Are you made of metal"},
                {"role": "assistant", "content": "Affirmative " + self.pronoun_str + "! I am made of metal"},
                {"role": "user", "content": "Is a mouse taller than a giraffe?"},
                {"role": "assistant", "content": "Negative " + self.pronoun_str + "! That is a very silly question."},
                {"role": "user", "content": "How many pounds are in a kilogram?"},
                {"role": "assistant", "content": "There are 2.20462 pounds in a kilogram, " + self.pronoun_str},
                {"role": "user", "content": "Is the sky blue?"},
                {"role": "assistant", "content": "Affirmative, " + self.pronoun_str},
                {"role": "user", "content": "Are you a teapot?"},
                {"role": "assistant", "content": "Negative " + self.pronoun_str + "! I am clearly not a teapot. You are a very silly human"},
                {"role": "user", "content": "Do cats climb trees?"},
                {"role": "assistant", "content": "Affirmative " + self.pronoun_str + "! Especially if I am chasing them."},
                {"role": "user", "content": "Do plants wear glasses?"},
                {"role": "assistant", "content": "Negative " + self.pronoun_str +"! Plants cannot see. You are an stupid person."}, 
        ]
        self.messages = []
        self.backstory = {"role": "system", "content": self.now_backstory}
        self.prompt = {"role": "user", "content": command}
        self.messages.append(self.backstory) # K9 backstory
        self.messages.extend(self.interactions_core) # A set of standard facts
        self.messages.extend(self.pronoun_interations) # A set of personalized answers based on gender and name
        self.messages.extend(self.interactions) # The interactions with this person to date
        self.messages.append(self.prompt) # The instruction that was given
        print("Response key:",openai.api_key, openai.organization)
        self.response_obj = openai.ChatCompletion.create(
            model = self.chat_model,
            messages = self.messages,
            temperature = 1.0,
            max_tokens = 100,
            top_p = 1.0,
            frequency_penalty = 0.0,
            presence_penalty = 0.0,
            )
        self.response = self.response_obj['choices'][0]['message']['content']
        self.response = self.response.strip('\n')
        self.response_msg = {"role": "assistant", "content": self.response}
        # now we need to add the prompt and the response to the interaction history
        self.interactions.append(self.prompt)
        self.interactions.append(self.response_msg)
        # now lets ensure the history doesn't get so long that it removes
        # the possibility of getting a response in 4096 tokens!
        self.length =  len(self.interactions)
        if self.length >= self.max_interactions_len:
            self.interactions = self.interactions[-self.max_interactions_len:]
        return self.intent, self.response