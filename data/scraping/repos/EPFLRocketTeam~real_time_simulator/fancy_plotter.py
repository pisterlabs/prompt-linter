from bokeh.plotting import figure, curdoc
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.io import show
from bokeh.layouts import gridplot, layout, row, column
from bokeh.models import CheckboxGroup, CustomJS, ColumnDataSource
from bokeh.models import Button, RadioGroup, FileInput, TextInput, RangeSlider, Slider, Panel, Tabs
from bokeh.themes import Theme

import rospy
import rospkg
import rosbag

from real_time_simulator.msg import FSM
from real_time_simulator.msg import State
from real_time_simulator.msg import Control
from real_time_simulator.msg import Sensor
from real_time_simulator.msg import Trajectory
from real_time_simulator.msg import Waypoint

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy import interpolate

import time

# Arrays with GNC data
simu_data = None
nav_data = None
feedback_data = None
control_data = None
target_data = None

# Display parameters
tStart = -1
tEnd = 40

target_iter = 0
point_spacing = 350

# Data type to display
simu = True
nav = False
horizon = False

def fill_full_state(bag, topic = ""):

  msg_count = bag.get_message_count(topic)
  np_data = np.zeros((14, msg_count))
  attitude = np.zeros((msg_count,4))
  i = 0
  for _, msg, t in bag.read_messages(topics=[topic]):
      new_attitude = msg.pose.orientation    
  
      attitude[i] = np.array([new_attitude.x, new_attitude.y, new_attitude.z, new_attitude.w])

      np_data[13, i] = t.to_sec()

      np_data[0, i] = msg.pose.position.x
      np_data[1, i] = msg.pose.position.y
      np_data[2, i] = msg.pose.position.z

      np_data[3, i] = msg.twist.linear.x
      np_data[4, i] = msg.twist.linear.y
      np_data[5, i] = msg.twist.linear.z

      np_data[9, i] = msg.twist.angular.x
      np_data[10, i] = msg.twist.angular.y
      np_data[11, i] = msg.twist.angular.z

      np_data[12, i] = msg.propeller_mass

      i = i+1

  r = R.from_quat(attitude)
  attitude_eul = r.as_euler('xyz', degrees=True)
  
  np_data[6:9, :] = np.transpose(attitude_eul)
  np_data[9:12, :] = np.rad2deg(np_data[9:12, :] )

  return np_data



def load_log_file(attr, old, new):
  global simu_data
  global nav_data
  global feedback_data
  global control_data
  global target_data

  rospack = rospkg.RosPack()
  bag = rosbag.Bag(rospack.get_path('real_time_simulator') + '/log/' + new)

  # Get first time message for synchronization
  for topic, msg, t in bag.read_messages(topics=['/fsm_pub']):
    if msg.state_machine != "Idle":
      time_init = t.to_sec()
      break
  
  simu_data = fill_full_state(bag, topic = "/rocket_state")

  nav_data = fill_full_state(bag, topic = "/kalman_rocket_state")


  control_data = np.zeros((5, bag.get_message_count('/control_pub')))
  i = 0
  for topic, msg, t in bag.read_messages(topics=['/control_pub']):
    control_data[0, i] = msg.force.x
    control_data[1, i] = msg.force.y
    control_data[2, i] = msg.force.z
    
    control_data[3, i] = msg.torque.z

    control_data[4, i] = t.to_sec()

    i = i+1

  feedback_data = np.zeros((5, bag.get_message_count('/control_measured')))
  i = 0
  for topic, msg, t in bag.read_messages(topics=['/control_measured']):
    feedback_data[0, i] = msg.force.x
    feedback_data[1, i] = msg.force.y
    feedback_data[2, i] = msg.force.z
    
    feedback_data[3, i] = msg.torque.z

    feedback_data[4, i] = t.to_sec()

    i = i+1

  # Guidance optimal trajectory
  target_positionX = []
  target_positionY = []
  target_positionZ = []
  target_speedZ = []
  target_prop_mass = []
  time_target = []
  thrust_target = []

  for topic, msg, t in bag.read_messages(topics=['/target_trajectory']):
    new_waypoint = msg.trajectory

    time_target.append([point.time for point in new_waypoint])
    target_positionX.append([point.position.x for point in new_waypoint])
    target_positionY.append([point.position.y for point in new_waypoint])
    target_positionZ.append([point.position.z for point in new_waypoint])
    target_speedZ.append([point.speed.z for point in new_waypoint])
    target_prop_mass.append([point.propeller_mass for point in new_waypoint])
    thrust_target.append([point.thrust for point in new_waypoint])
    
  bag.close()

  target_data = [target_positionZ, target_speedZ, target_prop_mass, thrust_target, time_target, target_positionX, target_positionY]
  print("Apogee: {}".format(max(simu_data[2])))

  # Synchronize time
  control_data[4] = control_data[4] - time_init
  simu_data[13] = simu_data[13] - time_init
  nav_data[13] = nav_data[13] - time_init
  feedback_data[4] = feedback_data[4] - time_init

  update_plot()

#select_target[::20,:] = True

def update_range(attr, old, new):
  global tStart
  global tEnd

  tStart = new[0]
  tEnd = new[1]

  update_plot()

def update_iteration(attr, old, new):
  global target_iter

  target_iter = new

  update_plot()

def update_nav_points(attr, old, new):
  global point_spacing

  point_spacing =  max(1, int(np.size(nav_data, 1)*(1 - np.log10(20*new+1)/3.3)))
  update_plot()

## ----------- Plot flight data ----------------------
doc = curdoc()
doc.theme = Theme(json={'attrs': {

    # apply defaults to Figure properties
    'Figure': {
        'outline_line_color': "DimGrey",
        'min_border': 10,
        'background_fill_color': "#FFFCFC",
        'plot_width':100, 

    },

    'Line': {
        'line_width': 2,
    },

    'Axis': {
        'axis_line_color': "DimGrey",
        
    },

    'Title': {
        'text_font_size': "12px",
        'text_line_height': 0.3,
        'align':'center',
    },

    'Legend': {
        'label_text_font_size': "10px",
        'background_fill_alpha': 0.5,
        'location': 'bottom_right',
    },
}})

# Create figures for plots
f_posXY = figure(title="Position [m]", title_location="left", x_axis_label='Time [s]')
f_posZ = figure(title="Position [m]", title_location="left", x_axis_label='Time [s]')
f_speedXY = figure(title="Speed [m/s]", title_location="left", x_axis_label='Time [s]')
f_speedZ = figure(title="Speed [m/s]", title_location="left", x_axis_label='Time [s]')
f_attitude = figure(title="Euler angle [°]", title_location="left", x_axis_label='Time [s]')
f_omega = figure(title="Angular rate [°/s]", title_location="left", x_axis_label='Time [s]')
f_thrust = figure(title="Main thrust [N]", title_location="left", x_axis_label='Time [s]')
f_force = figure(title="Side force [N]", x_axis_label='Time [s]')
f_mass = figure(title="Propellant mass [kg]", x_axis_label='Time [s]')

f_posZ_sel = figure(plot_width=600, plot_height=600)
f_speedZ_sel = figure(plot_width=600, plot_height=600)
f_thrust_sel = figure(plot_width=600, plot_height=600)


f_selectable = figure()
f_selectable.toolbar_location = "above"

f_posZ.toolbar_location = "above"
f_speedZ.toolbar_location = "below"
f_thrust.toolbar_location = "below"


# Create empty source for data
source_simu = ColumnDataSource(data=dict( t=[], 
                                          posX=[], posY=[], posZ=[],
                                          speedX=[], speedY=[], speedZ=[],
                                          attX=[], attY=[], attZ=[],
                                          omegaX=[], omegaY=[], omegaZ=[],
                                          mass = []
                                        ))

source_nav = ColumnDataSource(data=dict( t=[], 
                                          posX=[], posY=[], posZ=[],
                                          speedX=[], speedY=[], speedZ=[],
                                          attX=[], attY=[], attZ=[],
                                          omegaX=[], omegaY=[], omegaZ=[],
                                          mass = []
                                        ))

source_control = ColumnDataSource(data=dict(t=[], 
                                            thrust=[],
                                            forceX=[],
                                            forceY=[],
                                            torqueZ=[]))

source_feedback = ColumnDataSource(data=dict( t=[], 
                                              thrust=[],
                                              forceX=[],
                                              forceY=[],
                                              torqueZ=[]))

source_target = ColumnDataSource(data = dict( t=[],
                                              posZ=[],
                                              speedZ=[],
                                              mass=[],
                                              thrust=[]))


# Map simulation data to plots
f_posXY.line('t', 'posX', source=source_simu, color = "SteelBlue", legend_label="X")
f_posXY.line('t', 'posY', source=source_simu, color = "Coral", legend_label="Y")
f_posZ.line('t', 'posZ', source=source_simu, color = "Teal", legend_label="simu Z")
f_speedXY.line('t', 'speedX', source=source_simu, color = "SteelBlue", legend_label="X")
f_speedXY.line('t', 'speedY', source=source_simu, color = "Coral", legend_label="Y")
f_speedZ.line('t', 'speedZ', source=source_simu, color = "Teal", legend_label="simu Z")
f_attitude.line('t', 'attX', source=source_simu, color = "SteelBlue", legend_label="X")
f_attitude.line('t', 'attY', source=source_simu, color = "Coral", legend_label="Y")
f_attitude.line('t', 'attZ', source=source_simu, color = "Teal", legend_label="Z")
f_omega.line('t', 'omegaX', source=source_simu, color = "SteelBlue", legend_label="X")
f_omega.line('t', 'omegaY', source=source_simu, color = "Coral", legend_label="Y")
f_omega.line('t', 'omegaZ', source=source_simu, color = "Teal", legend_label="Z")
f_mass.line('t', 'mass', source=source_simu, color = "SeaGreen")

# Map navigation data to plots
f_posXY.scatter('t', 'posX', source=source_nav, marker = "+", line_dash='dashed', color = "SteelBlue", legend_label="X")
f_posXY.scatter('t', 'posY', source=source_nav, marker = "+", line_dash='dashed', color = "Coral", legend_label="Y")
f_posZ.scatter('t', 'posZ', source=source_nav, marker = "+", line_dash='dashed', color = "Teal", size=8, legend_label="est. Z")
f_speedXY.scatter('t', 'speedX', source=source_nav, marker = "+", line_dash='dashed', color = "SteelBlue", legend_label="X")
f_speedXY.scatter('t', 'speedY', source=source_nav, marker = "+", line_dash='dashed', color = "Coral", legend_label="Y")
f_speedZ.scatter('t', 'speedZ', source=source_nav, marker = "+", line_dash='dashed', color = "Teal", size=8, legend_label="est. Z")
f_attitude.scatter('t', 'attX', source=source_nav, marker = "+", line_dash='dashed', color = "SteelBlue", legend_label="X")
f_attitude.scatter('t', 'attY', source=source_nav, marker = "+", line_dash='dashed', color = "Coral", legend_label="Y")
f_attitude.scatter('t', 'attZ', source=source_nav, marker = "+", line_dash='dashed', color = "Teal", legend_label="Z")
f_omega.scatter('t', 'omegaX', source=source_nav, marker = "+", line_dash='dashed', color = "SteelBlue", legend_label="X")
f_omega.scatter('t', 'omegaY', source=source_nav, marker = "+", line_dash='dashed', color = "Coral", legend_label="Y")
f_omega.scatter('t', 'omegaZ', source=source_nav, marker = "+", line_dash='dashed', color = "Teal", legend_label="Z")
f_mass.scatter('t', 'mass', source=source_nav, marker = "+", line_dash='dashed', color = "SeaGreen")

# Map measured forces to plots
f_thrust.line('t', 'thrust', source=source_feedback, color = "FireBrick", legend_label="measured")
f_force.line('t', 'forceX', source=source_feedback, color = "SteelBlue", legend_label="X")
f_force.line('t', 'forceY', source=source_feedback, color = "Coral", legend_label="Y")
f_force.line('t', 'torqueZ', source=source_feedback, color = "Teal", legend_label="Z torque")

# Map controlled forces to plots
f_thrust.scatter('t', 'thrust', source=source_control, marker = "+", line_dash='dashed', color = "Orange", size=8, legend_label="command")
f_thrust.line('t', 'thrust', source=source_control, line_alpha=0.5, color = "Orange")
f_force.scatter('t', 'forceX', source=source_control, marker = "+", line_dash='dashed', color = "SteelBlue", legend_label="X")
f_force.scatter('t', 'forceY', source=source_control, marker = "+", line_dash='dashed', color = "Coral", legend_label="Y")
f_force.scatter('t', 'torqueZ', source=source_control, marker = "+", line_dash='dashed', color = "Teal", legend_label="Z torque")

# Map target from guidance to plots
f_thrust.line('t', 'thrust', source=source_target, line_alpha=0.5, line_width = 3, color="Orange")
f_posZ.line('t', 'posZ', source=source_target, line_alpha=0.5, line_width = 3, color="Teal")
f_posXY.line('t', 'posX', source=source_target, line_alpha=0.5, line_width = 3, color="SteelBlue")
f_posXY.line('t', 'posY', source=source_target, line_alpha=0.5, line_width = 3, color="Coral")
f_speedZ.line('t', 'speedZ', source=source_target, line_alpha=0.5, line_width = 3, color="Teal")
f_mass.line('t', 'mass', source=source_target, line_alpha=0.5, line_width = 3, color="SeaGreen")

# Selectable plot
tab1 = Panel(child=f_posZ, title="Altitude [m]")
tab2 = Panel(child=f_thrust, title="Main thrust [N]")
tab3 = Panel(child=f_speedZ, title="Speed [m/s]")
select_tabs = Tabs(tabs=[tab1, tab2, tab3],  width = 800, height = 600)

# Create Checkbox to select type of data to display
LABELS = ["Simulation", "Navigation", "Horizon"]
check_plot_type = CheckboxGroup(labels=LABELS, active=[0], margin = (20,5,20,30), background="Gainsboro", height = 60, width = 120)

# Create widget to define file path
file_name = TextInput(margin = (20,5,70,30))
file_name.on_change('value', load_log_file)

# Create slider to change displayed time
range_slider = RangeSlider(start=-1, end=60, value=(-1,30), step=.1, title="Time", width = 800, height = 10)
range_slider.on_change('value', update_range)

# Create slider to change displayed guidance iteration
iteration_slider = Slider(start=0, end=100, value=0, step=1, title="Guidance iteration", width = 250, margin = (20,5,70,30))
iteration_slider.on_change('value', update_iteration)

# Create slider to change density of displayed navigation points
nav_slider = Slider(start=0, end=100, value=50, step=1, title="Navigation points density [%]", margin = (20,5,70,30))
nav_slider.on_change('value', update_nav_points)

file_input = FileInput()
file_input.on_change('filename', load_log_file)

# Create complete layout with all widgets
grid_plot = gridplot([[f_posXY, f_posZ, f_attitude], [f_speedXY, f_speedZ, f_omega], [f_mass, f_thrust, f_force]], plot_width=450, plot_height=350)
main_plot = column(range_slider, grid_plot)
param_col = column(check_plot_type, file_input, iteration_slider, nav_slider)

doc_layout = row(main_plot, param_col, select_tabs)
doc.add_root(doc_layout)


def update_plot():

  if simu and np.all(simu_data) != None and np.all(feedback_data) != None:

    select = np.logical_and(simu_data[13]>tStart, simu_data[13] <tEnd)
    select_feedback = np.logical_and(feedback_data[4]>tStart, feedback_data[4] <tEnd) 

    
    source_simu.data = dict(t=simu_data[13][select],
                            posX=simu_data[0][select],
                            posY=simu_data[1][select],
                            posZ=simu_data[2][select],
                            speedX=simu_data[3][select],
                            speedY=simu_data[4][select],
                            speedZ=simu_data[5][select],
                            attX=simu_data[6][select],
                            attY=simu_data[7][select],
                            attZ=simu_data[8][select],
                            omegaX=simu_data[9][select],
                            omegaY=simu_data[10][select],
                            omegaZ=simu_data[11][select],
                            mass = simu_data[12][select])

    source_feedback.data=dict(t=feedback_data[4][select_feedback], 
                              thrust=feedback_data[2][select_feedback],
                              forceX=feedback_data[0][select_feedback],
                              forceY=feedback_data[1][select_feedback],
                              torqueZ=feedback_data[3][select_feedback])

  else:
    source_simu.data=dict(t=[], 
                          posX=[], posY=[], posZ=[],
                          speedX=[], speedY=[], speedZ=[],
                          attX=[], attY=[], attZ=[],
                          omegaX=[], omegaY=[], omegaZ=[],
                          mass = []
                        )

    source_feedback.data=dict([], 
                          thrust=[],
                          forceX=[],
                          forceY=[],
                          torqueZ=[])

  if nav and np.any(nav_data) != None and np.any(control_data) != None:
    
    select_est = np.logical_and(nav_data[13]>tStart, nav_data[13] <tEnd)    
    select_control = np.logical_and(control_data[4]>tStart, control_data[4] <tEnd) 

    source_nav.data = dict(t=nav_data[13][select_est][::point_spacing],
                            posX=nav_data[0][select_est][::point_spacing],
                            posY=nav_data[1][select_est][::point_spacing],
                            posZ=nav_data[2][select_est][::point_spacing],
                            speedX=nav_data[3][select_est][::point_spacing],
                            speedY=nav_data[4][select_est][::point_spacing],
                            speedZ=nav_data[5][select_est][::point_spacing],
                            attX=nav_data[6][select_est][::point_spacing],
                            attY=nav_data[7][select_est][::point_spacing],
                            attZ=nav_data[8][select_est][::point_spacing],
                            omegaX=nav_data[9][select_est][::point_spacing],
                            omegaY=nav_data[10][select_est][::point_spacing],
                            omegaZ=nav_data[11][select_est][::point_spacing],
                            mass = nav_data[12][select_est][::point_spacing])

    source_control.data=dict(t=control_data[4][select_control], 
                            thrust=control_data[2][select_control],
                            forceX=control_data[0][select_control],
                            forceY=control_data[1][select_control],
                            torqueZ=control_data[3][select_control])

  else: 

    source_nav.data=dict( t=[], 
                          posX=[], posY=[], posZ=[],
                          speedX=[], speedY=[], speedZ=[],
                          attX=[], attY=[], attZ=[],
                          omegaX=[], omegaY=[], omegaZ=[],
                          mass = []
                        )
    source_control.data=dict(t=[], 
                          thrust=[],
                          forceX=[],
                          forceY=[],
                          torqueZ=[])

  if horizon and np.any(target_data) != None:

    target_path = np.array([data[target_iter] for data in target_data])
    select_target =  np.logical_and(target_path[4]>tStart, target_path[4] <tEnd)

    source_target.data = dict(t=target_path[4][select_target],
                              posX=target_path[5][select_target],
                              posY=target_path[6][select_target],
                              posZ=target_path[0][select_target],
                              speedZ=target_path[1][select_target],
                              mass=target_path[2][select_target],
                              thrust=target_path[3][select_target])

def check_plot_type_handler(new):
  global simu
  global nav
  global horizon

  simu = True if 0 in new else False
  nav = True if 1 in new else False
  horizon = True if 2 in new else False

  update_plot()

check_plot_type.on_click(check_plot_type_handler)

