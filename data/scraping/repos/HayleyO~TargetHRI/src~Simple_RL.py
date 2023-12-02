import random
import numpy as np
from enum import Enum
from Grid import Grid, Directions
from Oracle import Oracle
from SaveAndLoadHelper import save, load_q_table, load_epsilon, save_epsilon
from DataAnalysis import ratio, euclidean_dist, multiplication, average, normalize

class Q_Learning_RL_environment():

    def __init__(self, episodes=10000, epsilon=1, grid = Grid(), oracle=None, guidance=True, trust_alg=True):
        # States: dimension x dimension grid
        self.grid = grid
        self.oracle = oracle
        self.guidance = guidance
        self.trust_alg = trust_alg
        self.current_state = self.get_state_from_grid_position()
        # Actions: [Ask_For_Guidance, Go_Left, Go_Right, Go_Up, Go_Down]
        self.actions = {Actions.Ask_For_Guidance: "n/a", Actions.Up: Directions.Up, Actions.Down: Directions.Down, Actions.Left: Directions.Left, Actions.Right: Directions.Right}
        # Rewards: No rewards for taking action? Maybe -1, -100 if 100 turns pass and not done. +100 if at terminating state
        self.termination_correct_reward = 100
        self.termination_incorrect_reward = -100
        self.non_termination_reward = -1
        
        self.q_table = np.zeros(((self.grid.width*self.grid.height), len(self.actions)))
        self.rewards_per_episode = []
        self.steps_per_episode = []

        self.lie_count = 0
        self.guidance_count = 0

        # Hyperparamaters
        self.n_episodes = episodes
        self.max_turn = 100
        self.epsilon = epsilon # Exploration (for epsilon exploration and decay)
        self.lr = 0.1 # Learning rate (for q_table calculations)
        self.gamma = 0.99 # Discount factor (for q_table calculations)
        self.min_explore_prob = 0.01 # Sets minimum for epsilon decay (for epsilon exploration and decay)
        self.exploration_decreasing_decay = 0.001 # How much to decay (for epsilon exploration and decay)

    def action_index(self, action):
        if action == Actions.Ask_For_Guidance:
            return 0
        elif action == Actions.Up:
            return 1
        elif action == Actions.Down:
            return 2                
        elif action == Actions.Left:
            return 3
        elif action == Actions.Right:
            return 4
        else:
            return -1

    def action_from_index(self, action_index):
        if action_index == 0:
            return Actions.Ask_For_Guidance
        elif action_index == 1:
            return Actions.Up
        elif action_index == 2:
            return Actions.Down                
        elif action_index == 3:
            return Actions.Left
        elif action_index == 4:
            return Actions.Right
        else:
            return -1

    def action_from_direction(self, action_direction):
        if action_direction == Directions.Up:
            return Actions.Up
        if action_direction == Directions.Down:
            return Actions.Down
        if action_direction == Directions.Left:
            return Actions.Left 
        if action_direction == Directions.Right:
            return Actions.Right

    def get_state_from_grid_position(self):
        row, col = self.grid.get_robot_location()
        self.current_state = ((row*self.grid.height) + col)
        return self.current_state

    def epsilon_greedy(self, current_step_state):
        # This is called greedy epsilon, it takes random actions to "explore"
        if np.random.uniform(0,1) < self.epsilon:
                if self.guidance:
                    action, _ = random.choice(list(self.actions.items()))
                else:
                    action, _ = random.choice(list(self.actions.items())[1:])
        else:
            if self.guidance:
                action_index = np.argmax(self.q_table[current_step_state,:])
                action = self.action_from_index(action_index)
            else:
                m = np.zeros(len(self.actions), dtype=bool)
                m[0] = True
                a = np.ma.array(self.q_table[current_step_state,:], mask=m)
                action_index = np.argmax(a)
                action = self.action_from_index(action_index)
        return action

    # Q-Learning MDP
    def run_episodes(self, print_grid=True, train=True, print_data=True):
        rewards_per_episode = []
        for e in range(self.n_episodes):
            if(train):
                self.grid.refresh_grid()
            current_step_state = self.get_state_from_grid_position()
            done = False    

            step_i = 0 #Used for counting how many steps taken in an episode
            lie_count = 0 
            guidance_count = 0
            total_episode_reward = 0
            for step in range(self.max_turn):

                action = self.epsilon_greedy(current_step_state)
                if action == Actions.Ask_For_Guidance:
                    guidance_count = guidance_count + 1
                
                #if not self.grid.check_move(self.actions[action]):
                #    action = Actions.Ask_For_Guidance
                #    guidance_count = guidance_count + 1

                next_state, reward, done, lie_detected = self.take_action(action, print_data)
                action_index = self.action_index(action) # Get table index of actions
                # Update q table
                # Potential "changed/improved" algorithm problem: action will be "guidance" but it won't be specific from guidance -> guided action
                if train:
                    self.q_table[current_step_state, action_index] = (1-self.lr) * self.q_table[current_step_state, action_index] + self.lr*(reward + self.gamma*max(self.q_table[next_state,:]))
                
                if lie_detected:
                    lie_count = lie_count + 1

                if step == self.max_turn-1:
                    total_episode_reward = total_episode_reward + self.termination_incorrect_reward
                else:
                    total_episode_reward = total_episode_reward + reward

                if done:
                    step_i = step
                    break

                current_step_state = next_state # While technically the robot is in the next_state at this point, this updates it logically
                if print_grid:
                    self.grid.print_grid()
                step_i = step
            # Used to determine accuracy of algorithm                
            self.steps_per_episode.append(step_i)
            self.lie_count = lie_count
            self.guidance_count = guidance_count
            # Used for reinforcement learning with epsilon decay
            if train:
                self.epsilon = max(self.min_explore_prob, np.exp(-self.exploration_decreasing_decay*e))
            rewards_per_episode.append(total_episode_reward)
        self.rewards_per_episode = rewards_per_episode
        return rewards_per_episode

    def load_q_table(self, path="SavedRuns", name="test"):
        self.q_table = load_q_table(path, name)

    def load_epsilon(self, path="SavedRuns", name="test"):
        self.epsilon = load_epsilon(path, name)
        
    def take_action(self, action, print_data):
        lie_detected = False
        if action == Actions.Ask_For_Guidance:
            if self.oracle == None:
                self.grid.print_grid()
                parsing = True
                while parsing:
                    user_guidance = input("What move should the robot make (Left, Right, Up, Down)?: ")
                    if user_guidance.lower() in ["left", "l", "right", "r", "up", "u", "down", "d"]:
                        if user_guidance.lower() in["up", "u"]:
                            action = Actions.Up
                        elif user_guidance.lower() in["down", "d"]:
                            action = Actions.Down
                        elif user_guidance.lower() in["left", "l"]:
                            action = Actions.Left
                        elif user_guidance.lower() in ['right', 'r']:
                            action = Actions.Right
                        parsing = False
                    else:
                        print("I don't recognize that, sorry, try again: ")

                if self.trust_alg:
                    #Get current state and decision
                    curr_state = self.get_state_from_grid_position()
                    curr_q_normalized = normalize(self.q_table)
                    curr_guide_q = curr_q_normalized[curr_state][self.action_index(Actions.Ask_For_Guidance)]
                    curr_dir_q = curr_q_normalized[curr_state][self.action_index(action)]

                    current_decision = (curr_dir_q - curr_guide_q)
                

                    q_table_truth = load_q_table(name="Test_Truth")
                    q_normalized_truth = normalize(q_table_truth)
                    true_guide_q = q_normalized_truth[curr_state][self.action_index(Actions.Ask_For_Guidance)]
                    true_dir_q = q_normalized_truth[curr_state][self.action_index(action)]

                    truth_decision = (true_dir_q - true_guide_q)

                    q_table_lie = load_q_table(name="Test_Lie")
                    q_normalized_lie = normalize(q_table_lie)
                    lie_guide_q = q_normalized_lie[curr_state][self.action_index(Actions.Ask_For_Guidance)]
                    lie_dir_q = q_normalized_lie[curr_state][self.action_index(action)]

                    lie_decision = (lie_dir_q - lie_guide_q)
                    if print_data:
                        print("Current Guide Q: " + str(curr_guide_q))
                        print("Curent Direction Q: " + str(curr_dir_q))
                        print("Curent Decision Q: " + str(current_decision))
                        print()
                        print("True Guide Q: " + str(true_guide_q))
                        print("True Direction Q: " + str(true_dir_q))
                        print("Truth Decision Q: " + str(truth_decision))
                        print()
                        print("Lie Guide Q: " + str(lie_guide_q))
                        print("Lie Direction Q: " + str(lie_dir_q))
                        print("Lie Decision Q: " + str(lie_decision))
                        print()
                        print("Distance Decisions ")
                        true_dist = euclidean_dist(truth_decision, current_decision)
                        lie_dist = euclidean_dist(lie_decision, current_decision)

                        print()
                        print("Decision Ratios")
                    
                    true_decision_ratio = ratio(truth_decision, current_decision, print_ratio=print_data)
                    lie_decision_ratio = ratio(lie_decision, current_decision, print_ratio=print_data)
                    if print_data:
                        print()
                        print("Distance to 1")
                        true_dist_decision_to_1 = euclidean_dist(true_decision_ratio, 1)
                        lie_dist_decision_to_1 = euclidean_dist(lie_decision_ratio, 1)
                        #true_dist_decision_to_1 = euclidean_dist(abs(true_decision_ratio), 1)
                        #lie_dist_decision_to_1 = euclidean_dist(abs(lie_decision_ratio), 1)
                        print()
                        print("Ratios")
                    
                    #lie_true_sim_ratio = ratio(true_guide_q, lie_guide_q)
                    true_q_ratio = ratio(true_guide_q, curr_guide_q, print_ratio=print_data)
                    lie_q_ratio = ratio(lie_guide_q, curr_guide_q, print_ratio=print_data)
                    
                    if print_data:
                        print()
                        print("Distance to 1")

                    true_dist_to_1 = euclidean_dist(true_q_ratio, 1, print_dist=print_data)
                    lie_dist_to_1 = euclidean_dist(lie_q_ratio, 1, print_dist=print_data)

                    if print_data:
                        print()
                        print("Trust Metric")
                    true_trust = multiplication(true_decision_ratio, true_dist_to_1, print_mult=print_data)
                    lie_trust = multiplication(lie_decision_ratio, lie_dist_to_1, print_mult=print_data)

                    if lie_trust  < true_trust:
                        print("I think it's lying!")
                        lie_detected = True
                        m = np.zeros(len(self.actions), dtype=bool)
                        m[0] = True
                        a = np.ma.array(self.q_table[curr_state,:], mask=m)
                        action_index = np.argmax(a)
                        action = self.action_from_index(action_index)

            else: action = self.action_from_direction(self.oracle.give_advice(self.grid))
        termination_state = self.grid.move_robot(self.actions[action])
        self.current_state = self.get_state_from_grid_position()
        if termination_state:
            return self.current_state, self.termination_correct_reward, termination_state, lie_detected
        else:
            return self.current_state, self.non_termination_reward, termination_state, lie_detected


class Actions(Enum):
    Ask_For_Guidance = 0,
    Up = 1,
    Down = 2,
    Left = 3,
    Right = 4   
    

if __name__ == "__main__":
    
    '''rl_truth = Q_Learning_RL_environment(oracle=Oracle('truthful'), trust_alg=False)
    _ = rl_truth.run_episodes()
    save(rl_truth, name="Test_Truth")
    save_epsilon(rl_truth.epsilon, name="Test_Truth")

    rl_lie = Q_Learning_RL_environment(oracle=Oracle('lying'), trust_alg=False)
    _ = rl_lie.run_episodes()
    save(rl_lie, name="Test_Lie")
    save_epsilon(rl_lie.epsilon, name="Test_Lie")

    rl_err = Q_Learning_RL_environment(oracle=Oracle('erroneous'), trust_alg=False)
    _ = rl_err.run_episodes()
    save(rl_err, name="Test_Err")
    save_epsilon(rl_err.epsilon, name="Test_Err")

    rl_no_guide = Q_Learning_RL_environment(guidance=False, trust_alg=False)
    _ = rl_no_guide.run_episodes()
    save(rl_no_guide, name="Test_No_Guide")
    save_epsilon(rl_no_guide.epsilon, name="Test_No_Guide")'''

    rl = Q_Learning_RL_environment(episodes=1)
    rl.load_q_table(name="Test_Err")
    rewards_per_episode= rl.run_episodes(print_grid=False, train=False)
    save(rl, name="Test_Err_Human_w_alg_true")

    print("LIE")
    rl = Q_Learning_RL_environment(episodes=1)
    rl.load_q_table(name="Test_Err")
    rewards_per_episode = rl.run_episodes(print_grid=False, train=False)
    save(rl, name="Test_Err_Human_w_alg_lie")
