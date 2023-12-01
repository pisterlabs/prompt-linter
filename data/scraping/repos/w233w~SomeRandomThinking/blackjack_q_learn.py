from tqdm import tqdm
import random
import gym
from collections import defaultdict


class BlackJackQLearn:
    '''
    BlackJack reinforced learning using simplfied BlackJack. For more details:
    https://github.com/openai/gym/blob/dcd185843a62953e27c2d54dc8c2d647d604b635/gym/envs/toy_text/blackjack.py#L50
    '''

    def __init__(self):
        # BlackJack environment from OpenAI/gym.
        self.env = gym.make('Blackjack-v1', new_step_api=True)

        # Used on Q_learning
        self.q_table = defaultdict(lambda: {0: 0.0, 1: 0.0})

        # When the AI knows nothing, the weight of action is evenly distributed.
        self.weights = defaultdict(lambda: [0.5, 0.5])

        # Used to collect rewards
        self.rewards = defaultdict(lambda: {0: [0.0, 0], 1: [0.0, 0]})

        self.policy = None

    # Global function to calculate the average of a list
    def _avg(self, lst):
        return sum(lst) / len(lst)

    def stream_avg(self, lst, new_val):
        if len(lst) != 2:
            raise ValueError('Error')
        avg, count = lst
        new_count = count + 1
        return [(avg * count + new_val) / new_count, new_count]

    def _generate_an_episode(self):
        # Used to store an episode
        episode = []

        state = self.env.reset()

        done = False
        while not done:

            # Take in an action
            action = random.choices(population=[0, 1], weights=self.weights[state], k=1)[0]

            # Interact with the environment to generate rewards and next states
            next_state, reward, done, *_ = self.env.step(action)

            # Store state, action, and reward to episode
            episode.append([state, action, reward])

            # Update state
            state = next_state

        return episode

    def _reversely_traversal_episode(self, episode):
        # Init the reward
        R = 0.0

        # Episode reverse traversal
        episode.reverse()
        for t in range(len(episode)):
            # Get state, action, and reward from each round in an episode
            state, action, reward = episode[t]

            # Reward function
            # R = R + γ * reward
            # γ = 1 in this case because every game is independent. So ignored here.
            R += reward

            # Add R into the collection of rewards.
            self.rewards[state][action] = self.stream_avg(self.rewards[state][action], R)

            # Updating Q-table
            self.q_table[state][action] = self.rewards[state][action][0]

            # The Winning decision has higher chance be used on the next episode
            good_choise = max(self.q_table[state], key=self.q_table[state].get)
            self.weights[state][good_choise] = 0.8
            self.weights[state][1 - good_choise] = 0.2

    def _monte_carlo(self, iters):
        for i in tqdm(range(iters)):
            # Generate episode
            episode = self._generate_an_episode()

            self._reversely_traversal_episode(episode)

    # gym.Env.BlackJackEnv only accept two input actions: 0 & 1 (False & True）
    # O represent stand and 1 represent hit.
    # Here we convert from {state: [0's reward, 1's reward]} to {state: 0 or 1} 
    def build_model(self):
        act = lambda dic: max(dic, key=dic.get)
        self.policy = {key: act(val) for key, val in self.q_table.items()}

    def fit(self, iters=100000):
        self._monte_carlo(iters)
        self.build_model()
        return self.policy

    # TEST
    # A single round of game
    def sample_simulation(self):
        init_state = self.env.reset()
        current_state = init_state

        done = False
        while not done:
            # Using policy get from AI to play the game, and collect the result.
            action = self.policy[current_state]
            current_state, reward, done, *_ = self.env.step(action)

        # Only collect initial state
        return init_state, reward

    def policy_check(self, iters):
        # Result
        result_set = {}
        for iter in tqdm(range(iters)):

            # Get init state of a game and it's result.
            state, reward = self.sample_simulation()

            # Store result
            if state in result_set:
                result_set[state].append(reward)
            else:
                result_set[state] = [reward]

        return result_set

    # Used to summarize win/tie/loss rate from policy_check() by state.
    def over_all_rate(self, game_result):
        rate = {}
        for result_state, result in game_result.items():
            win_rate = result.count(1.0) / len(result)
            tie_rate = result.count(0.0) / len(result)
            loss_rate = result.count(-1.0) / len(result)
            rate[result_state] = [win_rate, tie_rate, loss_rate]
        return rate

    # Print win and loss rate.
    def win_loss_rate(self, game_result):
        rate = self.over_all_rate(game_result)

        # result[0] is where win_rate in each state.
        win_rate = self._avg([result[0] for result in rate.values()])
        loss_rate = self._avg([result[2] for result in rate.values()])

        return win_rate, loss_rate

    def test(self, iters=100000):
        # Check policy and get policy win/loss rate.
        overall_ratio = self.policy_check(iters)
        return self.win_loss_rate(overall_ratio)


if __name__ == '__main__':
    a = BlackJackQLearn()
    policy = a.fit(100000)
    win_rate, loss_rate = a.test(100000)
    print("Win  Rate: " + str(win_rate))
    print("Loss Rate: " + str(loss_rate))
