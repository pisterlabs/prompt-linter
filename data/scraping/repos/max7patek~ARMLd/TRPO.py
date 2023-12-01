from TensorForceAdapter import default_environment as env

#from tensorforce.contrib.openai_gym import OpenAIGym
from tensorforce.agents import TRPOAgent
from tensorforce.execution import Runner
from tensorforce.core.explorations import EpsilonAnneal

from datetime import datetime

def main(max_timesteps):
    max_episodes = None
    #max_timesteps = 86400000000*days

    network_spec = [
        #dict(type='flatten'),
        dict(type='dense', size=11, activation='tanh'),
        #dict(type='dense', size=20, activation='tanh'),
        #dict(type='dense', size=32, activation='tanh'),
    ]

    exploration = dict(type='epsilon_decay', timesteps=max_timesteps)

    summarizer = dict(
        directory="./models/"+str(datetime.now()).replace(' ', ''),
        steps=10000,
        seconds=None,
        labels=[
            #'rewards',
            #'actions',
            'inputs',
            'gradients',
            'configuration',
        ],
        meta_dict=dict(
            description='July 2: Trying 11 node hidden layer.',
            layers=str(network_spec),
            timesteps=max_timesteps,
            exploration=exploration,
        ),
    )

    agent = TRPOAgent(
        states=env.states,
        actions=env.actions,
        network=network_spec,
        actions_exploration=exploration,
        #summarizer=summarizer,
        #batch_size=64
    )

    runner = Runner(agent, env)

    report_episodes = 1

    #global prev
    global prev
    prev = 0

    def episode_finished(r):
        global prev
        if r.episode % report_episodes == 0:
            #print("Finished episode {ep} after {ts} timesteps".format(ep=r.episode, ts=r.timestep-prev))
            #print("Episode reward: {}".format(r.episode_rewards[-1]))
            print(r.episode_rewards[-1])
        prev = r.timestep
        #print("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))
        return True

    print("Starting {agent} for Environment '{env}'".format(agent=agent, env=env))

    runner.run(num_episodes=max_episodes, num_timesteps=max_timesteps, max_episode_timesteps=None, episode_finished=episode_finished)

    agent.save_model(directory='./results/TRPO/'+str(datetime.now()).replace(' ', '')+'/model')

    runner.close()

    print("Learning finished. Total episodes: {ep}".format(ep=runner.episode))

if __name__ == '__main__':
    from sys import argv
    assert len(argv) == 2, 'input total num timesteps as command line argument'
    main(int(argv[1]))
