#!/usr/bin/env python 
# register the task_env and import the task_env
from gym.envs.registration import register
from gym import envs

def RegisterOpenAI_ROS_Env(task_env, max_episode_steps=10000):
    """
    Registers all the ENVS supported in OpenAI ROS. This way we can load them with variable limits.
    Here is where you have to PlACE YOUR NEW TASK ENV, to be registered and accessble.
    return: False if the Task Env wasnot registered, True if it was.
    """

    #########################################################
    # MovingCube Task-RObot Envs
    
    result = True

    # Cubli Moving Cube
    if task_env == 'CartPoleStayUp-v0':
        register(
            id=task_env,
            entry_point='openai_ros.task_envs.cartpole_stay_up.stay_up:CartPoleStayUpEnv',
            max_episode_steps=max_episode_steps,
        )
        # We have to import the Class that we registered so that it can be found afterwards in the Make
        from openai_ros_task_envs.cartpole_stay_up import stay_up

    elif task_env == 'MyTurtleBot2Wall-v0':

        register(
            id=task_env,
            entry_point='openai_ros.task_envs.turtlebot2.turtlebot2_wall:TurtleBot2WallEnv',
            max_episode_steps=max_episode_steps,
        )
        # import our training environment
        from openai_ros.task_envs.turtlebot2 import turtlebot2_wall
        
    elif task_env == 'MyTurtleBot2Maze-v0':
        register(
            id=task_env,
            entry_point='openai_ros.task_envs.turtlebot2.turtlebot2_wall:TurtleBOt2WallEnv',
            max_episode_steps=max_episode_steps,
        )
        #import our training environemnt
        from openai_ros.task_envs.turtlebot2 import turtlebot2_wall

    elif task_env == 'TurtleBot3World-v0':
        register(
            id=task_env,
            entry_point='openai_ros.task_env.turtlebot3_world:TurtleBot3WorldEnv',
            max_episode_steps=max_episode_steps,
        )
        #import our training environment 
        from openai_ros.task_envs.turtlebot3 import turtlebot3_world
    
    elif task_env == 'WamvNavTwoSetsBuoys-v0':

        register(
            id=task_env,
            entry_point='openai_ros.task_envs.wamv.wamv_nav_twosets_buoys:WamvNavTwoSetsBuoysEnv',
            max_episode_steps=max_episode_steps,
        )

        # import our training environment
        from openai_ros.task_envs.wamv import wamv_nav_twosets_buoys

    # Add here your Task Envs to be registered
    else:
        result = False
    
    ###############################################################################
    
    if result:
        # We check that it was really registered
        supported_gym_envs = GetAllRegisteredGymEnvs()
        # print("REGISTERED GYM ENVS ===>" + str(supported_gym_envs))
        assert (task_env in supported_gym_envs), "The task_Robot_ENV given is not Registered ==>" + \
            str(task_env)
    
    return result

def GetAllRegisteredGymEnvs():
    """
    REturns a list of all the registered Envs in the system
    return EX: ['Copy-v0', 'RepeatCopy-v0', 'ReversedAddition-v0', ... ]
    """

    all_envs = envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]

    return env_ids