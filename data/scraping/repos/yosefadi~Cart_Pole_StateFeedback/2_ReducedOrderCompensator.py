"""
PSKM - Reduced Order Compensator
Author:
    - Yosef Adi Sulistyo
    - Andreas Ryan C.K. 
    - Bonaventura Riko K.D.
    - Rafli Priyo Utomo
    - Muhammad Bagus H.

References:   https://pages.jh.edu/piglesi1/Courses/454/Notes6.pdf
Implemented in python with numpy
"""
# Check Python version
# Requires Python 3.6 or later
import sys
if sys.version_info < (3,6,0):
    print("Please use python 3.6.0 or higher")
    sys.exit(1)

# Version Checking Function
def versiontuple(v):
    return tuple(map(int, (v.split("."))))

# Try using Gymnasium (updated fork of OpenAI Gym) for the environment
# if not installed, fallback to OpenAI Gym Standard Module
try:
    import gymnasium as gym
    print("Using Gymnasium API.")
    legacy_api = False
except ImportError:
    import gym
    print("Gymnasium not installed. Using OpenAI Gym API instead.")
    if versiontuple(gym.__version__) < (0,23,0):
        print("Enabling compat API for OpenAI Gym version < 0.23.0")
        legacy_api = True
    else:
        legacy_api = False
    pass

# Import necessary module
import numpy as np
import control
import matplotlib.pyplot as plt

# Environment Variable
l = 0.5
mp = 0.1
mc = 1.0
g = 9.8
dt = 0.02  # from openai gym docs

# get environment
if legacy_api == True:
    env = gym.make('CartPole-v0').unwrapped
    env.seed(1)
    obs = env.reset()
else:
    env = gym.make('CartPole-v0', render_mode="human").unwrapped
    obs, info = env.reset(seed=1)

reward_threshold = 200
reward_total = 0

# System State Space Equation
A = np.array([[0, 1, 0, 0],
              [0, 0, -mp*(mp * (g-l) + mc*g)/((mc+mp)*((4/3) * mc + (1/3) * mp)), 0],
              [0, 0, 0, 1],
              [0, 0, (mp*(g-l) + mc * g)/(l*((4/3) * mc + (1/3) * mp)), 0]])

B = np.array([[0],
              [(1/(mc + mp) - mp/((mc + mp) * ((4/3) * mc + (1/3) * mp)))],
              [0],
              [(-1/(l * ((4/3) * mc + (1/3) * mp)))]])

C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

# reorganize matrices
Ar = np.array(A, copy=True)
Ar[[0, 1, 2, 3]] = Ar[[0, 2, 1, 3]]
Ar[:, [1, 2]] = Ar[:, [2, 1]]

Br = np.array(B, copy=True)
Br[[0, 1, 2, 3]] = Br[[0, 2, 1, 3]]

Cr = np.array(C, copy=True)
Cr[:,[1,2]] = Cr[:,[2,1]]

# partitioned matrices
# a = available states
# u = unavailable states
Aaa = Ar[:2,:2]
Aau = Ar[:2,2:]
Aua = Ar[2:,:2]
Auu = Ar[2:,2:]
Ba = Br[:2]
Bu = Br[2:]

# compute statenum
statenum = A.shape[0]

# desired poles
P = np.array([-0.5+0.5j, -0.5-0.5j, -21+0.25j, -21-0.25j])
Pt = 2 * P[2:]

# compute regulator and observer gains
K = control.place(A, B, P)
L = control.place(np.transpose(Auu), np.transpose(Aau), Pt)
L = np.transpose(L)

def compute_reduced_observer(x, x_hat, y, xcc, u):
    xcopy = np.array(x, copy=True)
    xa = np.empty([2,])
    xa[[0]] = xcopy[[0]]
    xa[[1]] = xcopy[[2]]
    print("xa: ", xa)

    x_hat_copy = np.array(x_hat, copy=True)
    xu_hat = np.empty([2,])
    xu_hat[[0]] = x_hat_copy[[1]]
    xu_hat[[1]] = x_hat_copy[[3]]
    print("xu_hat: ", xu_hat)

    xcc_dot = (Auu - L@Aau)@xu_hat + (Aua - L@Aaa)@y + (Bu - L@Ba)@u
    xcc = xcc + xcc_dot*dt
    xu_hat = xcc + L@y
    
    x_hat_new = np.concatenate((xa, xu_hat))
    x_hat_new[[2,1]] = x_hat_new[[1,2]]
    return xcc,x_hat_new
    
def apply_state_controller(x):
    u = -K@x   # u = -Kx
    if u > 0:
        action = 1
    else:
        action = 0 
    return action, u

obs_hat = np.zeros(4,)
xcc = np.zeros(2,)
u_array = []
x_array = []
x_dot_array = []
x_hat_dot_array = []
theta_array = []
theta_deg_array = []
theta_dot_array = []
theta_hat_dot_array = []
theta_dot_deg_array = []
theta_hat_dot_deg_array = []
t_array = []

for i in range(1000):
    # time logging
    t = i*dt
    t_array.append(t)

    env.render()

    # states data logging
    print("obs_hat: ", obs_hat)
    print("obs: ", obs)
    x_array.append(obs[0])
    x_dot_array.append(obs[1])
    theta_array.append(obs[2])
    theta_dot_array.append(obs[3])
    x_hat_dot_array.append(obs_hat[1])
    theta_hat_dot_array.append(obs_hat[3])

    # MODIFY THIS PART
    action, force = apply_state_controller(obs_hat)
    print("u:", force)

    # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
    clip_force = np.clip(force, -10, 10)
    abs_force = np.abs(float(clip_force))

    # log absolute force for plotting
    u_array.append(abs_force)

    # change magnitute of the applied force in CartPole
    env.force_mag = abs_force

    # apply action
    if legacy_api == True:
        obs, reward, done, info = env.step(action)
    else:
        obs, reward, done, truncated, info = env.step(action)
    
    # compute output y
    y = C@obs

    # compute observer state
    xcc,obs_hat = compute_reduced_observer(obs, obs_hat, y, xcc, clip_force)
    print("obs_hat: ", obs_hat)

    print()
    reward_total = reward_total+reward
    if done or reward_total == reward_threshold:
        print(f'Terminated after {i+1} iterations.')
        print("reward: ", reward_total)
        
        u_avg = np.around(np.mean(u_array),3)
        print("force_avg: ", u_avg, "N")

        for i in range(len(theta_array)):
            theta_deg_array.append(np.rad2deg(theta_array[i]))
            theta_dot_deg_array.append(np.rad2deg(theta_dot_array[i]))
            theta_hat_dot_deg_array.append(np.rad2deg(theta_hat_dot_array[i]))

        # plot 
        subplots = []
        for i in range(statenum):
            fig, ax = plt.subplots()
            subplots.append(ax)

        subplots[0].plot(t_array, x_array)
        subplots[0].set_title("x")
        subplots[0].set_xlabel("time (s)")
        subplots[0].set_ylabel("x")

        subplots[1].plot(t_array, x_dot_array, '-b', label="x_dot")
        subplots[1].plot(t_array, x_hat_dot_array, '--r', label="x_hat_dot")
        subplots[1].set_title("x dot")
        subplots[1].set_xlabel("time (s)")
        subplots[1].set_ylabel("dx/dt")
        
        subplots[2].plot(t_array, theta_deg_array)
        subplots[2].set_title("theta")
        subplots[2].set_xlabel("time (s)")
        subplots[2].set_ylabel("degree")

        subplots[3].plot(t_array, theta_dot_deg_array, '-b', label="theta_dot")
        subplots[3].plot(t_array, theta_hat_dot_deg_array, '--r', label="theta_hat_dot")
        subplots[3].set_title("theta dot")
        subplots[3].set_xlabel("time (s)")
        subplots[3].set_ylabel("dtheta/dt (deg/s)")

        if legacy_api == True:
            obs = env.reset()
        else:
            obs, info = env.reset()
        break

env.close()
plt.show(block=True)
