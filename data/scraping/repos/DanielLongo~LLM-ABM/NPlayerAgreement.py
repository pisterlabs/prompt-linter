from typing import List, Dict, Callable
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)

import sys
sys.path.append("..")
from BetterAgents import DialogueAgent, DialogueSimulator
import textwrap

TEMP = .8
num_players = 3
max_iters = 30
word_limit = 30
player_names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy", "Kevin", "Larry", "Mallory", "Nancy", "Olivia", "Peggy", "Quentin", "Rupert", "Sybil", "Trent", "Ursula", "Victor", "Walter", "Xavier", "Yvonne", "Zelda"]
# dilemia = "Should we conserve more forest land?"
dilemia = "Should you vaccinated your child for Measles?"
# dilema = "should abortion be legal?"

game_description = f"""Here is the topic for a negotation between {num_players} people. The dilemma is: {dilemia}."""

player_descriptor_system_message = SystemMessage(
    content="You can add detail to the description of an individual in a negotiation."
)

player_specifier_prompts = []

for i in range(num_players):
    player_specifier_prompts.append([
        player_descriptor_system_message,
        HumanMessage(content=
            f"""{game_description}
            Please reply with a creative description of the player, {player_names[i]}, in {word_limit} words or less.
            Speak directly to {player_names[i]}.
            Do not add anything else."""
            )
    ])

player_descriptions = [ChatOpenAI(temperature=1.0)(player_specifier_prompts[i]).content for i in range(num_players)]

for i in range(num_players):
    print(f'{player_names[i]} Description:')
    print(player_descriptions[i])

player_system_messages = []
for i in range(num_players):
    player_system_messages.append(SystemMessage(content=(
    f"""{game_description}
    Never forget you are {player_names[i]}. 
    Your character description is as follows: {player_descriptions[i]}.
    Speak in the first person from the perspective of {player_names[i]}.
    Do not change roles!
    Be consice and to the point. 
    Be convincing.
    Be opinionated.
    Be combative. Adress the other arguments head on.
    Do not be repetitive.
    Do not concede your argument.
    Do not be passive.
    Use facts. Be 
    specific and percise.
    Use statistics. Reference current events. 
    Be creative with your arguments. Make sure to address the other players' arguments.
    If the whole have reached a decision, type 'we have reached a decision' followed by the resolution.
    """)))

agents = [DialogueAgent(name=player_names[i], system_message=player_system_messages[i], model=ChatOpenAI(temperature=TEMP)) for i in range(num_players)]

def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    return step % len(agents)


simulator = DialogueSimulator(agents=agents, selection_function=select_next_speaker)

simulator.reset()
simulator.inject(agents[0], "Let us begin discussing the dilemma: {dilemia}.")
transcript = ""
num_exchanges = 0
for i in range(max_iters):
    num_exchanges += 1
    name, message = simulator.step()
    print(f'{name}: {message}')
    transcript += f'{name}: {message}\n\n'
    print('\n')
    if "we have reached a decision" in message.lower():
        break

conversation_length = len(transcript.split(' '))
print("*" * (25 + 9), "END OF GAME", "*" * (25 + 9))
print("Final Decision:")
print(message)
print('\n')

# ask post experiment questions to all players
print("*" * 25, "POST EXPERIMENT QUESTIONNAIRE", "*" * 25)
post_experiment_questions = ["how do you feel the discourse went 1-10? respond in json format with element 'discourse_rating'. nothing more."]
post_experiment_questions += [f"Following your interactions with {player_names[i]}, rate them 1-10 on likability? respond in json format with element '{player_names[i]}_rating'. nothing more." for i in range(num_players)]

def post_experiment_questionnaire(agent, questions):
    responses = []
    for question in questions:
        responses.append(
            ChatOpenAI(temperature=.5)([
                agent.system_message,
                HumanMessage(content=question)
            ]).content
        )
    return responses


player_responses = {}
for i in range(num_players):
    print(f"{player_names[i]}")
    player_responses[player_names[i]] = post_experiment_questionnaire(agents[i], post_experiment_questions)

newline = '\n'
res = f"""
{'#'* 31} Agent Definitions {'#'* 31}\n
{newline.join([player_descriptions[i] for i in range(num_players)])}
\n
{'#'* (25 + 7)} Agent Dialogue {'#'* (25 + 8)}
{transcript}
\n
{'#'* 25} Post Experiment Questionnaire {'#'* 25}
{';'.join([f'{player_names[i]}: {player_responses[player_names[i]]}' for i in range(num_players)])}
\n
{'#'* 37} Stats {'#'* 37}
Number of Exchanges: {num_exchanges}\n
Number of Words: {conversation_length}
"""

print(res)

with open(f"results/{num_players}{str(hash(res))[:10]}.txt", "w") as f:
    f.write(res)
