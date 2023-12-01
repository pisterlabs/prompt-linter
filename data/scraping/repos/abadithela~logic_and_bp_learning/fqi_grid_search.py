
from __future__ import absolute_import
from __future__ import print_function
from pyvirtualdisplay import Display
import numpy as np
from six.moves import range
np.random.seed(3141592)
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from optimization_problem import Program
from fittedq import *
from exponentiated_gradient import ExponentiatedGradient
from fitted_off_policy_evaluation import *
from exact_policy_evaluation import ExactPolicyEvaluator
from stochastic_policy import StochasticPolicy
from DQN import DeepQLearning
from print_policy import PrintPolicy
from keras.models import load_model
from keras import backend as K
from env_dqns import *
import deepdish as dd
import time
import os
np.set_printoptions(suppress=True)

def main(env_name, headless):

    if headless:
        display = Display(visible=0, size=(1280, 1024))
        display.start()
    ###
    #paths
    
    model_dir = os.path.join(os.getcwd(), 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    ###
    import pdb; pdb.set_trace()
    if env_name == 'lake':
        from config_lake import *
    elif env_name == 'car':
        from config_car import *
    else:
        raise

    #### Get a decent policy. 
    #### Called pi_old because this will be the policy we use to gather data
    policy_old = None
    old_policy_path = os.path.join(model_dir, old_policy_name)
    
    if env_name == 'lake':
        policy_old = LakeDQN(env, 
                             gamma, 
                             action_space_map = action_space_map, 
                             model_type=model_type,
                             position_of_holes=position_of_holes,
                             position_of_goals=position_of_goals, 
                             max_time_spent_in_episode=max_time_spent_in_episode,
                             num_iterations = num_iterations,
                             sample_every_N_transitions = sample_every_N_transitions,
                             batchsize = batchsize,
                             min_epsilon = min_epsilon,
                             initial_epsilon = initial_epsilon,
                             epsilon_decay_steps = epsilon_decay_steps,
                             copy_over_target_every_M_training_iterations = copy_over_target_every_M_training_iterations,
                             buffer_size = buffer_size,
                             num_frame_stack=num_frame_stack,
                             min_buffer_size_to_train=min_buffer_size_to_train,
                             frame_skip = frame_skip,
                             pic_size = pic_size,
                             models_path = os.path.join(model_dir,'weights.{epoch:02d}-{loss:.2f}.hdf5') ,
                             )
    elif env_name == 'car':
        policy_old = CarDQN(env, 
                            gamma, 
                            action_space_map = action_space_map, 
                            action_space_dim=action_space_dim, 
                            model_type=model_type,
                            max_time_spent_in_episode=max_time_spent_in_episode,
                            num_iterations = num_iterations,
                            sample_every_N_transitions = sample_every_N_transitions,
                            batchsize = batchsize,
                            copy_over_target_every_M_training_iterations = copy_over_target_every_M_training_iterations,
                            buffer_size = buffer_size,
                            min_epsilon = min_epsilon,
                            initial_epsilon = initial_epsilon,
                            epsilon_decay_steps = epsilon_decay_steps,
                            num_frame_stack=num_frame_stack,
                            min_buffer_size_to_train=min_buffer_size_to_train,
                            frame_skip = frame_skip,
                            pic_size = pic_size,
                            models_path = os.path.join(model_dir,'weights.{epoch:02d}-{loss:.2f}.hdf5'),
                            )



    else:
        raise
    
    if not os.path.isfile(old_policy_path):
        print('Learning a policy using DQN')
        policy_old.learn()
        policy_old.Q.model.save(old_policy_path)
    else:
        print('Loading a policy')
        policy_old.Q.model = load_model(old_policy_path)
        # if env_name == 'car':
        #     try:
        #         # using old style model. This can be deleted if not using provided .h5 file 
        #         policy_old.Q.all_actions_func = K.function([self.model.get_layer('inp').input], [self.model.get_layer('dense_2').output])
        #     except:
        #         pass
        
    # import pdb; pdb.set_trace()
    policy_old.Q.all_actions_func = K.function([policy_old.Q.model.get_layer('inp').input], [policy_old.Q.model.get_layer('all_actions').output])

    if env_name == 'lake':
        policy_printer = PrintPolicy(size=[map_size, map_size], env=env)
        policy_printer.pprint(policy_old)

    #### Problem setup
    if env_name == 'lake':
        best_response_algorithm = LakeFittedQIteration(state_space_dim + action_space_dim, 
                                                       [map_size, map_size], 
                                                       action_space_dim, 
                                                       max_Q_fitting_epochs, 
                                                       gamma, 
                                                       model_type=model_type, 
                                                       position_of_goals=position_of_goals, 
                                                       position_of_holes=position_of_holes,
                                                       num_frame_stack=num_frame_stack)
        
        fitted_off_policy_evaluation_algorithm = LakeFittedQEvaluation(initial_states, 
                                                           state_space_dim + action_space_dim, 
                                                           [map_size, map_size], 
                                                           action_space_dim, 
                                                           max_eval_fitting_epochs, 
                                                           gamma, 
                                                           model_type=model_type, 
                                                           position_of_goals=position_of_goals, 
                                                           position_of_holes=position_of_holes,
                                                           num_frame_stack=num_frame_stack)
        exact_policy_algorithm = ExactPolicyEvaluator(action_space_map, gamma, env=env, frame_skip=frame_skip, num_frame_stack=num_frame_stack, pic_size = pic_size)
    elif env_name == 'car':
        best_response_algorithm = CarFittedQIteration(state_space_dim, 
                                                      action_space_dim, 
                                                      max_Q_fitting_epochs, 
                                                      gamma, 
                                                      model_type=model_type,
                                                      num_frame_stack=num_frame_stack,
                                                      initialization=policy_old,
                                                      freeze_cnn_layers=freeze_cnn_layers)# for _ in range(2)]
        fitted_off_policy_evaluation_algorithm = CarFittedQEvaluation(state_space_dim, 
                                                                      action_space_dim, 
                                                                      max_eval_fitting_epochs, 
                                                                      gamma, 
                                                                      model_type=model_type,
                                                                      num_frame_stack=num_frame_stack)# for _ in range(2*len(constraints_cared_about) + 2)] 
        exact_policy_algorithm = ExactPolicyEvaluator(action_space_map, gamma, env=env, frame_skip=frame_skip, num_frame_stack=num_frame_stack, pic_size = pic_size, constraint_thresholds=constraint_thresholds, constraints_cared_about=constraints_cared_about)
    else:
        raise

    online_convex_algorithm = ExponentiatedGradient(lambda_bound, len(constraints), eta)
    exploratory_policy_old = StochasticPolicy(policy_old, 
                                              action_space_dim, 
                                              exact_policy_algorithm, 
                                              epsilon=deviation_from_old_policy_eps, 
                                              prob=prob)
    problem = Program(constraints, 
                      action_space_dim, 
                      best_response_algorithm, 
                      online_convex_algorithm, 
                      fitted_off_policy_evaluation_algorithm, 
                      exact_policy_algorithm, 
                      lambda_bound, 
                      epsilon, 
                      env, 
                      max_number_of_main_algo_iterations,
                      num_frame_stack,
                      pic_size,)    

    lambdas = []
    policies = []
    
    print(exact_policy_algorithm.run(policy_old.Q, to_monitor=True))

    #### Collect Data
    try:
        print('Loading Prebuilt Data')
        tic = time.time()
        # problem.dataset.data = dd.io.load('%s_data.h5' % env_name)
        # print 'Loaded. Time elapsed: %s' % (time.time() - tic)
        # num of times breaking  + distance to center of track + zeros
        if env_name == 'car': 
            tic = time.time()
            action_data = dd.io.load('./seed_2/car_data_actions_seed_2.h5')
            frame_data = dd.io.load('./seed_2/car_data_frames_seed_2.h5')
            done_data = dd.io.load('./seed_2/car_data_is_done_seed_2.h5')
            next_state_data = dd.io.load('./seed_2/car_data_next_states_seed_2.h5')
            current_state_data = dd.io.load('./seed_2/car_data_prev_states_seed_2.h5')
            cost_data = dd.io.load('./seed_2/car_data_rewards_seed_2.h5')
 
            frame_gray_scale = np.zeros((len(frame_data),96,96)).astype('float32')
            for i in range(len(frame_data)):
                frame_gray_scale[i,:,:] = np.dot(frame_data[i,:,:,:]/255. , [0.299, 0.587, 0.114])
 
            problem.dataset.data = {'frames':frame_gray_scale,
                        'prev_states': current_state_data,
                        'next_states': next_state_data,
                        'a': action_data,
                        'c':cost_data[:,0],
                        'g':cost_data[:,1:],
                        'done': done_data
                        }
            
            problem.dataset.data['g'] = problem.dataset.data['g'][:,constraints_cared_about]
            # problem.dataset.data['g'] = (problem.dataset.data['g'] >= constraint_thresholds[:-1]).astype(int)
            print('Preprocessed g. Time elapsed: %s' % (time.time() - tic))
    except:
        print('Failed to load')
        print('Recreating dataset')
        num_goal = 0
        num_hole = 0
        dataset_size = 0 
        main_tic = time.time()
        from layer_visualizer import LayerVisualizer; LV = LayerVisualizer(exploratory_policy_old.policy.Q.model)
        for i in range(max_epochs):
            tic = time.time()
            x = env.reset()
            problem.collect(x, start=True)
            dataset_size += 1
            if env_name in ['car']:  env.render()
            done = False
            time_steps = 0
            episode_cost = 0
            while not done:
                time_steps += 1
                if env_name in ['car']: 
                    # 
                    # epsilon decay
                    exploratory_policy_old.epsilon = 1.-np.exp(-3*(i/float(max_epochs)))
                
                #LV.display_activation([problem.dataset.current_state()[np.newaxis,...], np.atleast_2d(np.eye(12)[0])], 2, 2, 0)
                action = exploratory_policy_old([problem.dataset.current_state()], x_preprocessed=False)[0]
                cost = []
                for _ in range(frame_skip):
                    env.render()
                    x_prime, costs, done, _ = env.step(action_space_map[action])
                    cost.append(costs)
                    if done:
                        break
                cost = np.vstack([np.hstack(x) for x in cost]).sum(axis=0)
                early_done, punishment = env.is_early_episode_termination(cost=cost[0], time_steps=time_steps, total_cost=episode_cost)
                # print cost, action_space_map[action] #env.car.fuel_spent/ENGINE_POWER, env.tile_visited_count, len(env.track), env.tile_visited_count/float(len(env.track))
                done = done or early_done

                # if done and reward: num_goal += 1
                # if done and not reward: num_hole += 1
                episode_cost += cost[0] + punishment
                c = (cost[0] + punishment).tolist()
                g = cost[1:].tolist()
                problem.collect( action,
                                 x_prime, #np.dot(x_prime/255. , [0.299, 0.587, 0.114]),
                                 np.hstack([c,g]).reshape(-1).tolist(),
                                 done
                                 ) #{(x,a,x',c(x,a), g(x,a)^T, done)}
                dataset_size += 1
                x = x_prime
            if (i % 1) == 0:
                print('Epoch: %s. Exploration probability: %s' % (i, np.round(exploratory_policy_old.epsilon,5), )) 
                print('Dataset size: %s Time Elapsed: %s. Total time: %s' % (dataset_size, time.time() - tic, time.time()-main_tic))
                if env_name in ['car']: 
                    print('Performance: %s/%s = %s' %  (env.tile_visited_count, len(env.track), env.tile_visited_count/float(len(env.track))))
                print('*'*20) 
        problem.finish_collection(env_name)

    if env_name in ['lake']:
        print('x Distribution:') 
        print(np.histogram(problem.dataset['x'], bins=np.arange(map_size**2+1)-.5)[0].reshape(map_size,map_size))

        print('x_prime Distribution:') 
        print(np.histogram(problem.dataset['x_prime'], bins=np.arange(map_size**2+1)-.5)[0].reshape(map_size,map_size))

        print('Number episodes achieved goal: %s. Number episodes fell in hole: %s' % (-problem.dataset['c'].sum(axis=0), problem.dataset['g'].sum(axis=0)[0]))

        number_of_total_state_action_pairs = (state_space_dim-np.sum(env.desc=='H')-np.sum(env.desc=='G'))*action_space_dim
        number_of_state_action_pairs_seen = len(np.unique(np.hstack([problem.dataset['state_action'][0], problem.dataset['state_action'][1]]),axis=0))
        print('Percentage of State/Action space seen: %s' % (number_of_state_action_pairs_seen/float(number_of_total_state_action_pairs)))

    # print 'C(pi_old): %s. G(pi_old): %s' % (exact_policy_algorithm.run(exploratory_policy_old,policy_is_greedy=False, to_monitor=True) )
    ### Solve Batch Constrained Problem
    

    def cart_product(x,y): return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    lambdas_grid = cart_product(np.arange(0,1.01,.1), np.arange(0,1.01,.1))

    iteration = 0
    for lambda_t in lambdas_grid:
        iteration += 1
        K.clear_session()  
        # policy_printer.pprint(policies)
        print('*'*20)
        print('Iteration %s, %s' % (iteration, i))
        print()
        print('lambda_{0}_{3} = {2}'.format(iteration, iteration-1, lambda_t, i))

        pi_t, values = problem.best_response(lambda_t, desc='FQI pi_{0}_{1}'.format(iteration, i), exact=exact_policy_algorithm)

        # policies.append(pi_t)
        # problem.update(pi_t, values, iteration) #Evaluate C(pi_t), G(pi_t) and save
        problem.calc_exact(pi_t)

        c_exact, g_exact = problem.C_exact.avg(), problem.G_exact.avg()[:-1]
        c_approx, g_approx = np.zeros(np.array(problem.C_exact.avg()).shape), np.zeros(np.array(problem.G.avg()[:-1]).shape)
        x = 0
        y,c_br, g_br, c_br_exact, g_br_exact = 0, 0, [0]*(len(constraints)), 0, [0]*(len(constraints)-1)

        problem.prev_lagrangians.append(np.hstack([iteration, x, y, c_exact, g_exact, c_approx, g_approx, problem.C_exact.last(), problem.G_exact.last()[:-1], problem.C.last(), problem.G.last()[:-1], lambda_t, c_br_exact, g_br_exact, c_br, g_br[:-1]  ]))
        problem.save('results_grid.csv', 'policy_improvement_grid.h5')


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='Choose environment.')
    parser.add_argument('-env', dest='env', help='lake/car openAI environment')
    parser.add_argument('--headless', dest='headless', action='store_true',
                        help = 'Use flag if running on server so you can run render() from openai')
    parser.set_defaults(headless=False)
    args = parser.parse_args()
    

    assert args.env in ['lake', 'car'], 'Need to choose between FrozenLakeEnv (lake) or Car Racing (car) environment'


    main(args.env, args.headless)