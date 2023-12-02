'''
 Author: Jin
 Tensorflow
 Build based on Gym.env
'''

# import gym
import numpy as np
import scipy.sparse as sp

class SOFAEnv(object):
    """
    Following the setting from openai gym, we have API methods
    step
    reset
    render? not sure
    close
    seed

    And we have attributes:
    action_space,
    obvervation_space? is this state?
    reward_range ??

    Consider dealing with all users
    """
    def __init__(self, config):
        super(SOFAEnv, self).__init__()
        self.config = config
        # config setting
        self.episode_length = int(self.config['EPISODE_LENGTH'])

        # load ratings file
        # self.ratings = np.loadtxt(fname=self.config['RATING_FILE']) 
        self.ratings = self.config['RATINGS']
        if self.config['RATING_TYPE'] == 'matrix':
            (self.num_users, self.num_items) = self.ratings.shape
        else:
            print("wrong rating file type")
            exit(1)
        
        # train and test split (to-do-list)

        self.action_space = np.arange(self.num_items)
        # if one item is not avalible for a user, just mask this slot, self.user_action_mask[u, j] = True
        self.user_action_mask = sp.dok_matrix((self.num_users, self.num_items), dtype=bool)

        # initialize the env
        self.click_probability()
        self.state = [[], []] # [items, feedbacks]
        # np.random.seed(2020)

    def reset(self, user_id):
        self.user_id = user_id # to-do-list: multiple users
        self.history_items = set()
        self.step_count = 0
        self.state = [[], []]

    
    def step(self, action):
        '''
        Input: Action (item_id or item_ids)
        Return: state, reward, done, {some info}
        '''
        # if action in self.state[0]: # to-do-list: consider to change it into unclick
        #     print("recommend repeated item")
        #     exit(0)
        # # if repeat item, click_flag is 0
        click_flag = self.get_respond(action)
        if action in self.state[0]:
            click_flag = 0
        
        self.history_items.add(action) # now state is history_items, to-do-list: user_id/info
        # if click_flag == 1:
        #     self.state.append(action)
        self.state[0].append(action)
        self.state[1].append(click_flag)
        self.step_count += 1
        done = False
        if self.step_count >= self.episode_length:
            done = True
        return (self.state, click_flag, done)
            

    def click_probability(self):
        # self.click_p = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
        self.click_p = [0, 0, 0, 0, 1., 1.]
    def get_rating(self, item_id):
        return self.ratings[self.user_id, item_id]
    def get_respond(self, item_id):
        r = self.get_rating(item_id)
        if np.random.rand() <= self.click_p[r]:
            return 1
        else:
            return 0


# def simulated_data(num_users=500, num_items=200):
#     ratings = np.random.randint(1,6, size=(num_users, num_items))
#     return ratings

def simulated_data(num_users=500, num_items=200):
    import os, pickle
    module_path = os.path.dirname(__file__)  
    filename = module_path + "/simulated_U" + str(num_users) + "_I_" + str(num_items) + '.pkl'
    if os.path.exists(filename):
        print("load from %s" % filename)
        (ratingM, item_vec) = pickle.load(open(filename, 'rb'))
        unique, counts = np.unique(ratingM, return_counts=True)
        print("the rating count of the simulator, \n", unique, '\n', counts, flush=True)
        return ratingM, item_vec

    n_dim = 5
    # np.random.seed(2020)
    user_vec = np.random.randn(num_users, n_dim)
    item_vec = np.random.randn(num_items, n_dim)
    def sigmoid(x):
        return (1.0 / (1.0 + np.exp(-x)))
    rating_matrix = np.clip(np.round(sigmoid(np.dot(user_vec, item_vec.T) *0.8-1. )*4)+1, 1, 5)
    # count
    unique, counts = np.unique(rating_matrix, return_counts=True)
    print("the rating count of the simulator, \n", unique, '\n', counts, flush=True)
    pickle.dump((np.array(rating_matrix, dtype=np.int32), item_vec), open(filename, 'wb'))
    print("save into %s" % filename)
    return np.array(rating_matrix, dtype=np.int32), item_vec


if __name__ == "__main__":
    simulated_data(10, 20)
    exit(0)
    config = dict()
    config['RATINGS'], _ = simulated_data()
    config['RATING_TYPE'] = 'matrix'
    config['EPISODE_LENGTH'] = 10

    sofa = SOFAEnv(config)
    action_space = sofa.num_items
    for u in range(20):
        # initial
        sofa.reset(u)
        history_items = set()
        while True:
            action = np.random.randint(action_space)
            while action in history_items:
                action = np.random.randint(action_space)
            history_items.add(action)
            (state, reward, done) = sofa.step(action)
            if done:
                print("\n%d" % u)
                print(sofa.history_items)
                print(sofa.state)
                break
    
