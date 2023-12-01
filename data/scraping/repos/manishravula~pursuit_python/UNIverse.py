import gym
import numpy as np
from gym.envs.registration import  register
#Universe implementation of the OPENAI GYM

#ACTIONS:
# 0 - LEFT
# 1 - DOWN
# 2 - RIGHT
# 3 - UP

#need to modify the following from openAIgym

"""
1) Set up alternative envs, one for react func and other for act function
2) Get state as a single number representing the location 
3) Set up env.step to go back to the current state, i.e like a reset.

Just realized I can't do slippery because of how MCTS is formulated.

So this is going to become deterministic.
"""



LEFT = 0
RIGHT = 2
DOWN = 1
UP = 3

register(id='FrozenLakeNotSlippery-v0', entry_point='gym.envs.toy_text:FrozenLakeEnv', kwargs={'map_name' : '4x4', 'is_slippery': False}, max_episode_steps=100,)




class universe():
    #Could be plug-and-play
    #A game should allow :
    #1) Tell all possible legal states
    #2) Evaluate if a state is terminal or not
    #3) If it is terminal, tell who won.

    def __init__(self):
        #TODO: set this to the beginning state of every-universe
        #design: This state will always give the turn to the AI agent.

        self.env = gym.make('FrozenLakeNotSlippery-v0')
        self.env.reset()

        self.env_dummy = gym.make('FrozenLakeNotSlippery-v0')
        self.env_dummy.reset()


        self.env_rewardcalc = gym.make('FrozenLakeNotSlippery-v0')
        self.env_rewardcalc.reset()

        #holds flattened map
        self.MAP = self.env.env.desc.reshape(-1)


        self.state = self.env.env.s
        self.action_space = np.arange(4)
        self.derive_actionlist()


    #hacky - THis is needed for the MCTS code to directly refer to the state

    @property
    def state(self):
        print('DUmmy')
        return int(self.env.env.s)
    @state.setter
    def state(self,s):
        self.env.env.s = s

    def get_state(self):
        return self.env.env.s



    def set_state(self,state):
        self.env.env.s =  state

    def create_world(self):
        world = universe()
        return world

    def get_actionsLegal(self,state):
        #First get all moves allowed.
        #Apply them to the game and get the possible states.
        #Always generates a list of new legal next-state's features possible.
        #In FL, all actions are valid at all points.
        return self.actionlist_good[state]


    def react(self,action_external,state,Transition=False):
        #use
        """
        :param action_external: action taken by the external agent.
        :param state: state from which the agent is taking the action.
        :param Transition: Should the world transition into that state, or just peek and tell us what the state is.
        :return state_next: Returns the state after reacting.
        """
        #This function gives the universe's reaction to a particular user action in a state.
        #This could be thought as the transition when the agent acts, pushing the universe into its turn-taking state.


        if not Transition:
            return self.get_stateNext(state,action_external)
        else:
            #Transition means you have to move to that state.
            self.env.env.s = state
            _,_,_,_ = self.env.step(action_external)
            return self.env.env.s


    def act(self,state,Transition=False):
        #USe original env
        """
        :param state: state from which the world should act
        :param Transition: Should the world transition into that state, or just peek and tell us what the state is.
        :return reward: The reward you get because the world took this particular transition
        """
        #This function gives the universe's response to a particular state when the turn is its.
        #This cold be thought of as universe's move.`

        #DO nothing, because the universe is just a dumb player now.
        self.state = state

        #have to return reward, but here it is fake, so nothing happens anyway
        return 0
    def get_stateDisplay(self,state):
        st = np.copy(self.MAP)
        st[state] = str(st[state]).lower()
        final_st = ''
        for row in st.reshape((4,4)):
            for ele in row:
                final_st+=str(ele).swapcase()
            final_st+='\n'
        return final_st


    def is_terminalstate(self,state):
        if self.MAP[state]=='H' or self.MAP[state]=='G':
            return True
        else:
            return False

    def get_reward(self,curr_state,action,next_state):
        """
        use only after acting, not after acting.
        Calculates reward when you transition from current_state to next_state by taking 'action'
        :param curr_state:
        :param action:
        :param next_state:
        :return:
        """
        #Use dummy env
        state_preserve = self.env_rewardcalc.env.s

        self.env_rewardcalc.env.s = curr_state
        observation,reward,_,_ = self.env_rewardcalc.step(action)

        self.env_rewardcalc.env.s = state_preserve
        return reward
        #return one if player 1 won #Whose action we are trying to build a tree for
        #return -1 if player 2 won, Whose actions are generated randomly.

    def get_stateNext(self,curr_state,action_curr):
        #use dummy env
        preserveState = self.env_dummy.env.s

        self.env_dummy.env.s = curr_state
        observation = self.env_dummy.step(action_curr)
        state = self.env_dummy.env.s

        self.env_dummy.env.s = preserveState

        return state #Some dummy

    def derive_actionlist(self):
        actionlist_good = []
        for state in range(15):
            new_states = []
            actionlist_new = []
            for action in self.action_space:
                newstate = self.get_stateNext(state,action)
                if newstate==state:
                    pass
                else:
                    actionlist_new.append(action)
            actionlist_good.append(actionlist_new)
        self.actionlist_good = actionlist_good


    def hash_action(self,action):
        """
        :action: n_agents length array of actions.

        possible actions for each agent are:
        1) move up - u
        2) move down - d
        3) move right -r
        4) move left - l
        5) load - a

        :return: no_agents sized string.
        """
        hash_action= ''
        for act in action:
            hash_action+=self.action_hashes[act]
        return hash_actio

    def unhash_action(self,hashed_action):
        """

        :param hashed_action: str of hashed actions
        :return: vector of actions
        """

        action_vector = []
        for char in hashed_action:
            action_vector.append(self.dict_actionIndex[char])
        return np.array(action_vector)

    def react(self,action):
        real_action = self.unhash_action(action)
        real_action = real_action[-1] #actionprob of the last MCTS agent.

        action_probs = np.zeros(5)
        action_probs[real_action]=1

        action_and_consequence = self.mcts_agent.behave_act(action_probs)
        self.mcts_agent.execute_action(action_and_consequence)

        pre_length = len(self.items)

        if np.any([agent.load for agent in self.agents]):
            self.update_foodconsumption()
        reward = len(self.items)-pre_length
        self.check_for_termination()

        if self.visualize:
            self.update_vis()

        #we return action itself because we assume a unique action-state relationship.
        #and it seems pointless to hash a state and unhash it
        return reward, action

    def act(self):
        agent_actions = []

        # retrieve what the agent wants to do
        for agent in self.agents:
            # Check what the agent wants to do
            action_probs = agent.behave()

            # retrieve the action.
            agent_action = agent.behave_act(action_probs)

            agent_actions.append(agent_action)

            # Approve the agent's action. This way, if agent moves further and is in
            # collision path with another agent, then this is not going to be aproble
            # as the other agent will plan accordingly.
            agent.execute_action(agent_action)

        pre_length = len(self.items)
        # See if there is any load operation.
        if np.any([agent.load for agent in self.agents]):
            self.update_foodconsumption()

        self.check_for_termination()

        reward = len(self.items) - pre_length
        if self.visualize:
            self.update_vis()

        #state just needs to be something unique.
        return reward, str(reward)


    def act_external(self,action):

        n_agents = len(self.agents)-1
        real_action = self.unhash_action(action)
        agent_actions = []
        for i in range(n_agents):
            # Check what the agent wants to d
            agent = self.agents[i]
            action_probs = np.ones(5)
            action_probs[real_action[i]]=1

            # retrieve the action.
            agent_action = agent.behave_act(action_probs)

            agent_actions.append(agent_action)

            # Approve the agent's action. This way, if agent moves further and is in
            # collision path with another agent, then this is not going to be aproble
            # as the other agent will plan accordingly.
            agent.execute_action(agent_action)

        pre_length = len(self.items)
        # See if there is any load operation.
        if np.any([agent.load for agent in self.agents]):
            self.update_foodconsumption()

        self.check_for_termination()

        reward = len(self.items) - pre_length
        if self.visualize:
            self.update_vis()

        # state just needs to be something unique.
        return reward, str(reward)

    def get_legalActions(self, turn):

        if turn is False:  # universe's turn
            validactionString_list = []
            for agent in self.agents[0:-1]:  # exclude MCTS agent
                validactionProb = agent.get_legalActionProbs()
                valid_actions = np.argwhere(validactionProb != 0).reshape(-1)
                validAction_string = ''
                for action in valid_actions:
                    validAction_string += self.action_hashes[action]
                validactionList.append(validAction_string)
            validactionString_list.append('n')
            all_actionStringVectors = itertools.product(*validactionString_list)
            for i in range(len(validactionString_list)):
                validactionString_list[i] = ''.join(validactionString_list[i])
            return validactionString_list
        else:
            validactionString_list = []
            default_string = ''.join(['n' for i in range(self.no_items - 1)])
            validactionProb = self.mcts_agent.get_legalActionProbs()
            valid_actions = np.argwhere(validactionProb != 0).reshape(-1)
            for action in valid_actions:
                validactionString_list.append(default_string + self.action_hashes[action])

            return validactionString_list

        ##Yet to finish

    def get_legalActions_N(self, turn):
        if turn in False:  # universe's turn
            tot_actions = 1
            for agent in self.agents[0:-1]:
                validactionProb = agent.get_legalActionProbs()
                n_actions = np.sum(validactionProb != 0)
                tot_actions *= n_actions
            return tot_actions

        else:
            agent = self.agents[-1]
            return np.sum(agent.get_legalActionProbs() != 0)

    def get_legalAction_random(self, turn):
        action_string = ''
        if turn is False:  # universe's turn
            for agent in self.agents[0:-1]:
                validactionProb = agent.get_legalActionProbs()
                randomaction = np.random.choice(np.where(validactionProb != 0)[0])
                action_string += self.action_hashes[randomaction]
            action_string += 'n'
        else:
            action_string += sum(['n' for i in range(self.no_agents - 1)])
            validactionProb = self.mcts_agent.get_legalActionProbs()
            randomaction = np.random.choice(np.where(validactionProb != 0)[0])
            action_string += self.action_hashes[randomaction]
        return action_string
