# General imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
import scienceplots
plt.style.use(['science','grid', 'no-latex'])
from scipy.signal import csd 

# MCSimPython library
from MCSimPython import simulator as sim
from MCSimPython import control as ctrl
from MCSimPython import guidance as ref
from MCSimPython import observer as obs
from MCSimPython import thrust_allocation as ta
from MCSimPython import waves as wave
from MCSimPython.utils import six2threeDOF, three2sixDOF, timeit, Rz
# Not yet pushed to init.py
from MCSimPython.observer.ltv_kf import LTVKF 
from MCSimPython.control.adaptiveFS import AdaptiveFSController
from MCSimPython.thrust_allocation.allocation import fixed_angle_allocator
from MCSimPython.simulator.thruster_dynamics import ThrusterDynamics
from MCSimPython.thrust_allocation.thruster import Thruster
from MCSimPython.vessel_data.CSAD.thruster_data import lx, ly, K

# Sim parameters --------------------------------------------------------
dt = 0.02
N = 20000
np.random.seed(1234)

# Vessel simulator --------------------------------------------------
vessel = sim.CSAD_DP_6DOF(dt = dt)


# External forces -----------------------------------------------------
# Current
U = 0.00
beta_u = np.deg2rad(180)
nu_cn = U*np.array([np.cos(beta_u), np.sin(beta_u), 0])

# Waves
N_w = 25                                            # Number of wave components
hs = 0.03                                           # Significant wave height
tp = 1.2                                            # Peak period
wp = 2*np.pi/tp                                     # Peak frequency
wmin = wp/2
wmax = 2.5*wp
dw = (wmax-wmin)/N_w
w_wave = np.linspace(wmin, wmax, N_w)               # Frequency range

# Spectrum
jonswap = wave.JONSWAP(w_wave)
freq, spec = jonswap(hs, tp, gamma=1.)              # PM spectrum

wave_amps = np.sqrt(2*spec*dw)                      # Wave amplitudes
eps = np.random.uniform(0, 2*np.pi, size=N_w)       # Phase 
wave_dir = np.deg2rad(180) * np.ones(N_w)           # Direction

# Wave loads
waveload = wave.WaveLoad(wave_amps, w_wave, eps, wave_dir, 
                            config_file=vessel._config_file) 



# Observer ----------------------------------------------------------
observer = LTVKF(dt, vessel._M, vessel._D, Tp=tp)
observer.set_tuning_matrices(
            np.array([
                [1e2,0,0,0,0,0],
                [0,1e3,0,0,0,0],
                [0,0,1e2*np.pi/180,0,0,0],
                [0,0,0,1e4,0,0],
                [0,0,0,0,1e4,0],
                [0,0,0,0,0,1e2]]), 
            np.array([
                [1e-0,0,0],
                [0,1e-3,0],
                [0,0,np.pi/180]]))

# Reference model -----------------------------------------------------
ref_model = ref.ThrdOrderRefFilter(dt, omega = [.25, .2, .2])   #
eta_ref = np.zeros((N, 3))                                      # Stationkeeping


# Controller ----------------------------------------------------------
N_adap = 0
N_theta = (6*N_adap + 3)
controller = AdaptiveFSController(dt, vessel._M, vessel._D, N = N_adap)

w_min_adap = 2*np.pi/20 
w_max_adap = 2*np.pi/2                           # Upper bound
controller.set_freqs(w_min_adap, w_max_adap, N_adap)

K1 = [20, 20, .1]
K2 = [60, 60, 1]
gamma_adap = np.ones((N_theta))*0.7

controller.set_tuning_params(K1, K2, gamma=gamma_adap)


# Thrust allocation --------------------------------------------------
thrust_dynamics = ThrusterDynamics()
thrust_allocation = fixed_angle_allocator()
for i in range(6):
    thrust_allocation.add_thruster(Thruster([lx[i],ly[i]],K[i]))



wave_realization = jonswap.realization(time=np.arange(0,N*dt,dt), hs = hs, tp=tp)

# Simulation ========================================================

storage = np.zeros((N, 91 + N_theta))

t_global = time()
for i in tqdm(range(N)):
    t = (i+1)*dt
    
    zeta= wave_realization[i]
    
    # Accurate heading measurement
    psi = vessel.get_eta()[-1]
    nu_cb = Rz(psi).T@nu_cn

    # Ref. model
    eta_d, eta_d_dot, eta_d_ddot = np.zeros(3), np.zeros(3) , np.zeros(3)                                                     # 3 DOF
    nu_d = Rz(psi).T@eta_d_dot

    # Wave forces
    tau_w_first = waveload.first_order_loads(t, vessel.get_eta())
    tau_w_second = waveload.second_order_loads(t, vessel.get_eta()[-1])
    tau_w = tau_w_first + tau_w_second

    # Controller
    time_ctrl = time()
    tau_cmd, bias_ctrl = controller.get_tau(observer.get_eta_hat(), eta_d,  observer.get_nu_hat(), eta_d_dot, eta_d_ddot, t, calculate_bias = True)
    time_ctrl2 = time() - time_ctrl

    theta_hat = controller.get_theta()

    # Thrust allocation - not used - SATURATION
    u, alpha = thrust_allocation.allocate(tau_cmd)
    tau_ctrl = thrust_dynamics.get_tau(u, alpha)

    # Measurement
    noise = np.concatenate((np.random.normal(0,.001,size=3),np.random.normal(0,.0002,size=3)))
    y = np.array(vessel.get_eta()) + noise

    # Observer
    observer.update(tau_ctrl, six2threeDOF(y), psi)

    # Calculate x_dot and integrate
    tau = three2sixDOF(tau_ctrl) + tau_w
    vessel.integrate(U, beta_u, tau)

    # Calculate simulator bias
    nu_cb_ext = np.concatenate((nu_cb, np.zeros(3)), axis=None)
    b_optimal = six2threeDOF((vessel._D)@nu_cb_ext + tau_w_second)
    b_optimal_ned = Rz(psi)@b_optimal
    gamma_adap = 0

    storage[i] = np.concatenate([t, vessel.get_eta(), vessel.get_nu(), eta_d, nu_d, y, tau_cmd, tau_w, 
                                 observer.get_x_hat(), eta_ref[i], bias_ctrl, tau_ctrl, time_ctrl2, tau_w_first, 
                                 tau_w_second, u, zeta, K1, K2, gamma_adap, b_optimal, b_optimal_ned, theta_hat], axis=None)

t_global = time() - t_global

# Post processing 
# ============================================================================================
t = storage[:,0]
eta = storage[:,1:7]
nu = storage[:, 7:13]
eta_d = storage[:, 13:16]
nu_d = storage[:, 16:19]
y = storage[:,19:25]
tau_cmd = storage[:, 25:28]
tau_w = storage[:, 28:34]
xi_hat = storage[:, 34:40]
eta_hat = storage[:, 40:43]
bias_hat = storage[:, 43:46]
nu_hat = storage[:, 46:49]
eta_ref = storage[:, 49:52]
bias = storage[:, 52:55]
tau_ctrl = storage[:, 55:58]
t_ctrl = storage[:, 58]
tau_w_first = storage[:, 59:65]
tau_w_second = storage[:, 65:71]
u = storage[:, 71:77]
zeta = storage[:,77]
k1 = storage[:,78:81]
k2 = storage[:, 81:84]
gamma_adap = storage[:, 84]
b_optimal_body = storage[:, 85:88]
b_optimal_ned = storage[:, 88:91]
theta_hat = storage[:,91:]


# Position plot ==============================================================================
fig, axs = plt.subplots(2, 3)
i_obs=0
for i in range(2):
    for j in range(3):
        DOF = j+1+3*i
        #axs[i,j].plot(t, y[:, DOF-1], label=r'$y$'+str(DOF))
        axs[i,j].plot(t, eta[:, DOF-1], label=r'$\eta$'+str(DOF))
        axs[i,j].grid()
        axs[i,j].set_xlim([0,dt*N])
        axs[i,j].set_xlabel('t [s]')
        axs[i,j].set_title(r'$\eta $'  + str(DOF))
        if i == 1:
            axs[i,j].set_ylabel('Angle [rad]')
        else:
            axs[i,j].set_ylabel('Disp [m]')  
        if DOF in [1,2,6]: 
            axs[i,j].plot(t, eta_hat[:,i_obs], label=r'$\eta_{obs} $ '+str(DOF))
            axs[i,j].plot(t, eta_d[:, i_obs], label=r'$\eta_{d}$'+str(DOF))
            i_obs+=1
        
        axs[i,j].legend( edgecolor="k")
plt.tight_layout()
plt.suptitle('Response', fontsize=32)

# Velocity ==============================================================================
fig, axs = plt.subplots(2, 3)
i_obs=0
for i in range(2):
    for j in range(3):
        DOF = j+1+3*i
        axs[i,j].plot(t, nu[:, DOF-1], label=r'$\nu$'+str(DOF))
        axs[i,j].grid()
        axs[i,j].set_xlim([0,dt*N])
        axs[i,j].set_xlabel('t [s]')
        axs[i,j].set_title(r'$\nu $'  + str(DOF))
        if i == 1:
            axs[i,j].set_ylabel('Angle vel. [rad/s]')
        else:
            axs[i,j].set_ylabel('Vel [m/s]')  
        if DOF in [1,2,6]: 
            axs[i,j].plot(t, nu_hat[:,i_obs], label=r'$\nu_{obs} $ '+str(DOF))
            axs[i,j].plot(t, nu_d[:,i_obs], label=r'$\nu_{d} $ '+str(DOF))
            i_obs+=1
        
        axs[i,j].legend(edgecolor="k")

plt.tight_layout()
plt.suptitle('Velocity', fontsize=32)


# North-East plot ==============================================================================
plt.figure(figsize=(8,8))
plt.plot(eta[:,1], eta[:,0], label='Trajectory', linewidth=2)
plt.plot(eta_d[:,1], eta_d[:,0], '-.', label='eta_d')
plt.grid()
plt.title('XY-plot')
plt.legend(loc="upper right", edgecolor="k")
plt.xlabel('East [m]')
plt.ylabel('North [m]')
plt.axis('equal')


# Control forces plot ==============================================================================
fig, axs = plt.subplots(3, 1)
plt.suptitle('Control Forces')
for i in range(3):
    axs[i].plot(t, tau_cmd[:,i], label=r'$\tau_{cmd}$ '+ str(i+1))
    axs[i].plot(t, tau_ctrl[:,i], label=r'$\tau_{ctrl}$ '+ str(i+1))
    axs[i].plot(t, bias[:,i], label='Disturbance estimation')
    axs[i].plot(t, bias_hat[:,i], label='Observer estimation')
    axs[i].legend()
    axs[i].set_title(r'$\tau$' + str(i+1))
plt.legend()


fig, axs = plt.subplots(3, 1)
plt.suptitle('Residual loads')
for i in range(3):
    axs[i].plot(t, bias[:,i], label='Bias from controller '+ str(i+1))
    axs[i].plot(t, bias_hat[:,i], label='Bias from observer '+ str(i+1))
    axs[i].plot(t, b_optimal_body[:,i], label='Residual load from simulator - BODY')
    axs[i].plot(t, b_optimal_ned[:,i], label='Residual load from simulator, NED')
    axs[i].legend()
    axs[i].set_title(r'$\tau$' + str(i+1))
plt.legend()



N_theta_plot = 2*N_adap + 1
theta_surge = theta_hat[-1, 0:N_theta_plot]
theta_sway = theta_hat[-1, N_theta_plot:2*N_theta_plot]
theta_yaw = theta_hat[-1, 2*N_theta_plot:]
theta_list = [theta_surge, theta_sway, theta_yaw]

fig, axs = plt.subplots(1, 1)
plt.suptitle('Adaptive gains')
for i in range(3):
    axs.plot(theta_hat[-1, 0+i*(N_theta_plot):(N_theta_plot)+i*(N_theta_plot)], label='DOF '+str(i+1))
    axs.plot(theta_list[i], label='check')
    axs.legend()
