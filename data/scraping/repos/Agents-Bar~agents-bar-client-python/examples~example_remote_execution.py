# Example of how to create an agent and an environment.
# Once we create these, we then communicate them (locally) so that agent interacts with the environment.
from agentsbar import Client, agents, environments, experiments
from agentsbar.types import DataSpace
from agentsbar.types import AgentCreate, EnvironmentCreate, ExperimentCreate
from agentsbar.utils import wait_until_active


# Define client to communicate with https://agents.bar. Make sure it's authenticated.
print("Initiate Agents Bar client")
client = Client()

# Create an environment. Simple one is "CartPole-v1" from OpenAI gym repo.
print("Create an initiate environment")
env_name = "CartPole"
env_create = EnvironmentCreate(name=env_name, image="agents-bar/env-gym", config={"gym_name": "CartPole-v1"})
environments.create(client, env_create=env_create)
wait_until_active(client, 'environment', env_name)

# Create an agent. Since environment is discrete we use DQN.
agent_name = "CartPoleAgent"
print("Create an initiate agent")
agent_config = {
    'obs_space': DataSpace(dtype='float', shape=(4,)),
    'action_space': DataSpace(dtype='int', shape=(1,), low=0, high=2),
}
agent_create = AgentCreate(name=agent_name, model='DQN', image='agents-bar/agent', config=agent_config)
agents.create(client, agent_create=agent_create)
wait_until_active(client, 'agent', agent_name)

# Create an Experiment which allows for Agent <-> Environment communication
exp_name = "CartPoleExperiment"
print("Create an initiate experiment")
experiment_create = ExperimentCreate(
    name=exp_name, agent_names=[agent_name], environment_names=[env_name], config={}, description="Testing experimnt on CartPole"
)
experiments.create(client, experiment_create)
wait_until_active(client, 'experiment', exp_name)

# One command allows to start the communication.
# After this the whole learning process is done in Agents Bar.
print("Starting learning")
experiments.start(client, exp_name)
