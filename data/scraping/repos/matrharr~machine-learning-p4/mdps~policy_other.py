import mdptoolbox.example
from hiive.mdptoolbox.mdp import PolicyIterationModified, PolicyIteration
from helpers.open_ai_convert import OpenAI_MDPToolbox
from helpers.plot_graphs import plot_discount, plot_rewards

disc = [0.1, 0.3, 0.5, 0.7, 0.9]
ep = [0.00099, 0.001, 0.005, 0.01, 0.03]

ex = OpenAI_MDPToolbox('FrozenLake-v0', False)
P = ex.P
R = ex.R
results = []
for d in disc:
    pi = PolicyIteration(
    # pi = PolicyIterationModified(
        P, # transitions
        R, # rewards
        d, # discount
        # epsilon=0.01,
        max_iter=1000,
    )
    pi.run()
    print('policy iteration value function:', pi.V)
    print('policy iteration iterations:', pi.iter)
    print('policy iteration time:', pi.time)
    print('policy iteration best policy:', pi.policy)
    results.append(pi)

plot_rewards(
    disc, results, 'Policy Iteration Discount/Rewards FrozenLake',
    'policy_iteration_discount_rewards_frozenlake', 'Discount'
)
results = []
for e in ep:
    pi = PolicyIteration(
    # pi = PolicyIterationModified(
        P, # transitions
        R, # rewards
        0.9, # discount
        # epsilon=e,
        max_iter=1000,
    )
    pi.epsilon = e
    pi.run()
    print('policy iteration value function:', pi.V)
    print('policy iteration iterations:', pi.iter)
    print('policy iteration time:', pi.time)
    print('policy iteration best policy:', pi.policy)
    results.append(pi)

plot_rewards(
    ep, results, 'Policy Iteration Epsilon/Rewards FrozenLake',
    'policy_iteration_epsilon_rewards_frozenlake', 'Epsilon'
)

print('----------------Best PI FrozenLake---------------')
pi = PolicyIteration(
    # pi = PolicyIterationModified(
        P, # transitions
        R, # rewards
        0.9, # discount
        max_iter=1000,
    )
pi.run()
print('policy iteration value function:', pi.V)
print('policy iteration iterations:', pi.iter)
print('policy iteration time:', pi.time)
print('policy iteration best policy:', pi.policy)
