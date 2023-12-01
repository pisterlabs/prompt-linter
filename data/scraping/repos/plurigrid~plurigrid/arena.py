from chatarena.agent import Player
from chatarena.backends import OpenAIChat
from chatarena.environments.conversation import Conversation
from chatarena.arena import Arena
import time

environment_description = "In a decentralized electricity market with distributed energy resources (DERs), operate a game-theoretic framework that efficiently manages energy supply while minimizing costs and maximizing profits for all participants. The framework should address network and physical constraints, evaluate the effectiveness of individual and collaborative strategies, and encourage coalition formation among agents such as electric vehicle owners, solar panel owners, and battery owners. Incorporate optimization algorithms, fuzzy logic, or cooperative game theory to facilitate effective collaboration and decision-making within the grid."

agent1 = Player(
    name="Electric Vehicle Owner",
    backend=OpenAIChat(),
    role_desc="As the owner of an electric vehicle, your role in the intelligent grid structure is to minimize the cost of charging your vehicle. Collaborate with other DER participants like solar panel owners and battery owners to form a coalition that benefits all parties involved. Use your influence and strategic thinking to optimize the power usage of your vehicle, maximize profits, and contribute to the overall efficiency of the grid.",
    global_prompt=environment_description,
)

agent2 = Player(
    name="Solar Panel Owner",
    backend=OpenAIChat(),
    role_desc="As a solar panel owner, your responsibility is to maximize the revenue generated from selling solar energy to the grid. Engage with other participants like electric vehicle owners and battery owners to create cooperative strategies that benefit all parties involved. Leverage the power of the intelligent grid and optimization algorithms to find effective collaborative strategies, resulting in increased profits and energy efficiency.",
    global_prompt=environment_description,
)

agent3 = Player(
    name="Battery Owner",
    backend=OpenAIChat(),
    role_desc="As a solar panel owner, your responsibility is to maximize the revenue generated from selling solar energy to the grid. Engage with other participants like electric vehicle owners and battery owners to create cooperative strategies that benefit all parties involved. Leverage the power of the intelligent grid and optimization algorithms to find effective collaborative strategies, resulting in increased profits and energy efficiency.",
    global_prompt=environment_description,
)

env = Conversation(player_names=[p.name for p in [agent1, agent2, agent3]])
arena = Arena(
    players=[agent1, agent2, agent3], environment=env, global_prompt=environment_description
)

# arena.launch_cli()
timestamp = time.time()
arena.run(num_steps=20)
arena.save_history(path=f"./plurigrid/ontology/arena_history_{timestamp}.json")
