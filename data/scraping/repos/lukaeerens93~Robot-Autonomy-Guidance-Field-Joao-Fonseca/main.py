# Import dependencies and libraries
import numpy as np
import math
import LIDAR        # You need to create this python file
import Over_Camera  # You need to create this python file
import IMU
import GUIDANCE
import Motor_input

# Define the initial state
x0, y0, th0, vx0, vy0, w0, t = 0, 0, 0, 0, 0, 0, 0
State0 = [x0, y0, th0, vx0, vy0, w0]
State0Mem = State0



# Tank Definitions
L, W = 0.2, 0.1          # Length and Width
R_wh = 0.05              # wheel radius
Max_size = 0.5*max(L, W)

Clearance_distance = 2
N_points_clearance = 30
Clearance_Target = Clearance_distance + Max_size/2

Belt_Length = 2 * math.pi * R_wh * L        # length of the belt in the wheels
dt = 0.01                                   # integration time
Target_Cruise_speed = 3.0                   # ratio to the maximum speed
# 0 no movement, 1 to target full speed when travelling in straight lines



# Define Hard Obstacles (Real Coordinates)
'''
This is where you need to change them to demonstrate flexibility with obstacle configuration
'''
#            Matlab Code:
# x_H_Obs = np.array([30 30 20 20 20 20 20 20 20 20 20]); y_H_Obs = [25 20 30 25 20 15 10 5 0 11 13];
H_Obs = np.array([
    [30, 25],
    [30, 20],
    [20, 30],
    [20, 25],
    [20, 20],
    [20, 15],
    [20, 10],
    [20, 5],
    [20, 0],
    [20, 11],
    [20, 13]
])
# Number of Hard Obstacles in the array
N_H_Obs = H_Obs.size

# Define Soft Obstacles (real coordinates)
#               Matlab Code:
# x_S_Obs=[+2 +4 +6 +8 +20 +12 +35 +20 +45]; y_S_Obs=[+5 +5 +5 +5 +5 +10 +15 +12.5 +5]
S_Obs = np.array([
    [2, 5],
    [4, 5],
    [6, 5],
    [8, 5],
    [20, 5],
    [12, 10],
    [35, 15],
    [20, 12.5],
    [45, 5]
])
# Number of Soft obstacles in the array
N_S_Obs = S_Obs.size


# Camera receives Pixels and outputs coordinates in meters. Scaling is Pixel per meter
Image_W, Image_L, Scaling = 5000, 3000, 100

# Target State
x_target, y_target = 40, 25
Target = [x_target, y_target]



# ---------------------------   Main Loop Starts Here   ------------------------------------
Log = [t, 0, 0, State0]    # Time, Motor Multiplers (Left, and then Right), State
Delta_Target = 0

# While distance to the target is smaller than what you can do with,
# the current speed you need to exit because you have arrived
#           Refer back to line 17 to figure out the variables
delta_x = State0[0] - x_target
delta_y = State0[1] - y_target
delta_x = math.pow(delta_x, 2)
delta_y = math.pow(delta_y, 2)

delta_v = math.pow(State0[3], 2) + math.pow(State0[4], 2)   # Euclidean Magnitude of Veloticy


while math.pow(delta_x + delta_y, 0.5) > (math.pow(delta_v), 0.5) * dt * 2:
    # ----------------------------------------------------------------------------------------
    # SENSOR INPUT: Get LIDAR readings
    [distance, bearing] = LIDAR(H_Obs, State0)       # lidarData[0] = distance, lidarData[1] = bearing


    # SENSOR INPUT: Get Over_Camera readings
    [S_Obs_measure, target_measure, position_measure] = Over_Camera(S_Obs,
                                                                    Image_W,
                                                                    Image_L,
                                                                    Scaling,
                                                                    Target,
                                                                    State0)

    # -----------------------------------------------------------------------------------------------
    # SENSOR INPUT: IMU readings
    [th_IMU, x_IMU, y_IMU, z_IMU, vx_IMU, vy_IMU, vz_IMU] = IMU(State0);


    # -----------------------------------------------------------------------------------------------
    # SENSOR FUSION: Here is where we decide what goes into the State update

    # Convert LIDAR input to real world coordinates
    for j in range(0, N_H_Obs,1):
        H_Obs[j][0] = State0[0] + distance[j] * math.cos(bearing[j])   # lidarData[1] = bearing
        H_Obs[j][1] = State0[1] + distance[j] * math.sin(bearing[j])   # lidarData[0] = distance


    # convert soft obstacles to real world coordinates
    S_Obs[0] = S_Obs_measure[:][0] * Scaling / Image_L          # S_Obs_measure[:][0] = x_S_Obs_measure
    S_Obs[1] = S_Obs_measure[:][1] * Scaling / Image_W          # S_Obs_measure[:][1] = y_S_Obs_measure


    # Convert target to real world coordinates to real world coordinates
    Target[0] = target_measure[0] * Scaling / Image_L
    Target[1] = target_measure[1] * Scaling / Image_W


    # Convert current position to real world coordinates to real world coordinates
    # (alternative use IMU or fuse both with Kalman filter)
    State0[0] = position_measure[0] * Scaling / Image_L
    State0[1] = position_measure[1] * Scaling / Image_W


    # Update rest of the state vector
    State0[2] = th_IMU
    State0[3] = vx_IMU
    State0[4] = vx_IMU
    State0[5] = vy_IMU

    # ------------------------------------------------------------------------------------------------------------------------
    # GUIDANCE FIELD:
    # In the simulation we compute for the whole space, but in the real thing we only compute at current location

    # Return only the field at current location
    [Guidance_Field] = GUIDANCE(Image_W,
                                Image_L,
                                Scaling,
                                Target,
                                H_Obs,
                                S_Obs,
                                State0,
                                Clearance_Target,
                                t)
    # !!!!!!!!!!!!!!!   Make sure that you get rid of the variables in the actual function as well !!!!!!!!!!!!

    # Atan2 is already confined by default inside[-pi, pi]
    th_target = math.atan2(Guidance_Field[1], Guidance_Field[0])

    # Need to contain Delta_Target to[-pi, pi]
    Delta_Target = th_target - State0[2]

    # Delta_Target = Delta_Target + 1 / 380 * 2 * pi;
    Delta_Target_old = Delta_Target

    if Delta_Target >= + math.pi
        Delta_Target = Delta_Target - 2 * math.pi


    if Delta_Target <= -math.pi
        Delta_Target = Delta_Target + 2 * math.pi;

    # print (str(Delta_Target_old) + " " + str(Delta_Target))

    # Get the multipliers
    [Motor_Left_Multiplier, Motor_Right_Multiplier] = Motor_input(Delta_Target)


    # Motors and its inverse will go here
    Motor_Left_Speed = Motor_Left_Multiplier * Target_Cruise_speed
    Motor_Right_Speed = Motor_Right_Multiplier * Target_Cruise_speed

