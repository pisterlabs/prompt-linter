from typing import List, Dict, Callable
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)

import sys
sys.path.append("..")
from BetterAgents import DialogueAgent, DialogueSimulator

buyer_name = "Bob"
seller_name = "George"
quest = "Find a fair price for a used car."
word_limit = 50 # word limit for task brainstorming

game_description = f"""Here is the topic for a negotation between two parties: {quest}.
        One individual is looking to buy the car: the buyer, {buyer_name}.
        One individual is looking to sell the car: the salenames, {seller_name}."""

player_descriptor_system_message = SystemMessage(
    content="You can add detail to the description of an individual in a negotiation.")

buyer_specifier_prompt = [
    player_descriptor_system_message,
    HumanMessage(content=
        f"""{game_description}
        Please reply with a creative description of the buyer, {buyer_name}, in {word_limit} words or less. 
        Speak directly to {buyer_name}.
        Do not add anything else."""
        )
]
buyer_description = ChatOpenAI(temperature=1.0)(buyer_specifier_prompt).content

seller_specifier_prompt = [
    player_descriptor_system_message,
    HumanMessage(content=
        f"""{game_description}
        Please reply with a creative description of the seller, {seller_name}, in {word_limit} words or less. 
        Speak directly to {seller_name}.
        Do not add anything else."""
        )
]
seller_description = ChatOpenAI(temperature=1.0)(seller_specifier_prompt).content

print('Buyer Description:')
print(buyer_description)
print('Seller Description:')
print(seller_description)

car_description = "The car is a 2012 Lexus LS 460 AWD with 27,000 miles."

buyer_system_message = SystemMessage(content=(
f"""{game_description}
Never forget you are the buyer, {buyer_name}, and I am the seller, {seller_name}. 
Your character description is as follows: {buyer_description}.
You will propose deals and offers and are trying to buy the car for as cheap as possible. {car_description}
Speak in the first person from the perspective of {buyer_name}.
For describing your own body movements, wrap your description in '*'.
Do not change roles!
Do not speak from the perspective of {seller_name}.
Do not add anything else.
Remember you are the buyer, {buyer_name}.
Stop speaking the moment you finish speaking from your perspective.
"""
))

seller_system_message = SystemMessage(content=(
f"""{game_description}
Never forget you are the seller, {seller_name}, and I am the buyer, {buyer_name}. 
Your character description is as follows: {seller_description}.
You will propose deals and offers and are trying to sell the car to me for as much as possible (be super aggressive with the buyer). {car_description}
Speak in the first person from the perspective of {seller_name}.
Do not change roles!
Do not speak from the perspective of {buyer_name}.
Do not add anything else.
Remember you are the seller, {seller_name}.
Stop speaking the moment you finish speaking from your perspective.
"""
))

quest_specifier_prompt = [
    SystemMessage(content="You can make a task more specific."),
    HumanMessage(content=
        f"""{game_description}
        
        You are the seller, {seller_name}.
        Please reply with the specified response in {word_limit} words or less. 
        Speak directly to the protagonist {buyer_name}.
        Do not add anything else."""
        )
]
specified_quest = ChatOpenAI(temperature=1.0)(quest_specifier_prompt).content

print(f"Original response:\n{quest}\n")
print(f"Detailed response:\n{specified_quest}\n")

buyer = DialogueAgent(name=buyer_name,
                     system_message=buyer_system_message, 
                     model=ChatOpenAI(temperature=0.2))
seller = DialogueAgent(name=seller_name,
                     system_message=seller_system_message, 
                     model=ChatOpenAI(temperature=0.2))

def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    idx = step % len(agents)
    return idx

max_iters = 6
n = 0

simulator = DialogueSimulator(
    agents=[seller, buyer],
    selection_function=select_next_speaker
)
simulator.reset()
simulator.inject(seller_name, specified_quest)
print(f"({seller_name}): {specified_quest}")
print('\n')

while n < max_iters:
    name, message = simulator.step()
    print(f"({name}): {message}")
    print('\n')
    n += 1