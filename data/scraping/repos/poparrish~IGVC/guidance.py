#!/usr/bin/env python
import math
import traceback

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, Vector3, TransformStamped, Transform, Point32
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud
from std_msgs.msg import String, Header, Int16
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from tf2_msgs.msg import TFMessage

import topics
from guidance import compute_potential
from guidance.attractor_placement import generate_path
from guidance.gps_guidance import dist_to_waypoint, calculate_gps_heading, current_angle
from guidance.potential_field import extract_repulsors, ATTRACTOR_THRESHOLD_MM
from mapping import MAP_SIZE_PIXELS, MAP_SIZE_METERS
from util import Vec2d, avg, to180
from util.rosutil import extract_tf, rx_subscribe, get_rotation

GUIDANCE_NODE = "GUIDANCE"
GUIDANCE_HZ = 10

#
# Drivetrain
#

FAST_SPEED = 1.8  # m/s
DIST_CUTOFF = 1500  # slow down once we're within 2m of something
SLOW_SPEED = 1.6
RAMP_SPEED = 2

MAX_ROTATION = 90  # max crabbing angle
MAX_TRANSLATION = 90  # max crabbing angle

control = None


def update_drivetrain(translation, rotation, speed):
    translation = max(-MAX_TRANSLATION, min(translation, MAX_TRANSLATION))
    rotation = max(-MAX_ROTATION, min(rotation, MAX_ROTATION))
    # speed, velocity_vector, theta_dot
    control.publish(Vector3(x=speed, y=translation, z=-rotation))


#
# Waypoints
#

GPS_BUFFER = 10  # buffer GPS messages to help with accuracy

FIRST_WAYPOINT_TOLERANCE = 1.5  # when to start tracking the first waypoint
WAYPOINT_TOLERANCE = 1  # precision in meters
RAMP_INCLINE_TOLERANCE = 5  # how many degrees of incline before we switch states

LAT_OFFSET = 0.0
LON_OFFSET = 0.0

WAYPOINTS = [
    (42.6790651932,-83.1949756404),  # north entry
    (42.6789723207,-83.1951336744),  # north ramp
    (42.6788536649,-83.1951441500),  # south ramp
    (42.6787645626, -83.1949480515)  # south entry
]


def reached_waypoint(num, gps_buffer, tolerance):
    (lat, lon) = WAYPOINTS[num - 1]
    distance = avg([dist_to_waypoint(msg, (lat + LAT_OFFSET, lon + LON_OFFSET))
                    for msg in gps_buffer])
    # state_debug.publish(str(distance))

    return distance < tolerance

def going_up(gps_buffer,tolerance):
    angle = avg([current_angle(loc) for loc in gps_buffer])
    return angle > tolerance

def going_down(gps_buffer,tolerance):
    angle = avg([current_angle(loc) for loc in gps_buffer])
    return angle < tolerance*-1

def going_flat(gps_buffer,tolerance):
    angle = avg([current_angle(loc) for loc in gps_buffer])
    return (angle < tolerance and angle > tolerance*-1)

#
# State machine
#

LINE_FOLLOWING = 'LINE_FOLLOWING'
WAYPOINT_TRACKING = 'WAYPOINT_TRACKING'

TRACKING_FIRST_WAYPOINT ='TRACKING_FIRST_WAYPOINT'#this can be beginning and end state
TRACKING_SECOND_WAYPOINT = 'TRACKING_SECOND_WAYPOINT'
TRACKING_THIRD_WAYPOINT = 'TRACKING_THIRD_WAYPOINT'
TRACKING_FOURTH_WAYPOINT = 'TRACKING_FOURTH_WAYPOINT'
CLIMBING_UP = 'CLIMBING_UP'
CLIMBING_DOWN = 'CLIMBING_DOWN'



DEFAULT_STATE = {
    'state': LINE_FOLLOWING,
    'speed': SLOW_SPEED,
    'tracking': 1
}
# DEFAULT_STATE = {
#     'state': WAYPOINT_TRACKING,
#     'speed': INITIAL_SPEED,
#     'tracking': 0
# }

heading_debug = rospy.Publisher(topics.POTENTIAL_FIELD, PoseStamped, queue_size=1)
path_debug = rospy.Publisher('guidance/path', Path, queue_size=1)
obstacle_debug = rospy.Publisher('guidance/obstacles', PointCloud, queue_size=1)


def compute_next_state(state, gps_buffer):
    """guidance state machine"""
    # rospy.loginfo(state)
    if state['state'] == TRACKING_FIRST_WAYPOINT:
        if reached_waypoint(1, gps_buffer, tolerance=FIRST_WAYPOINT_TOLERANCE):
            rospy.loginfo('Begin tracking second waypoint')
            return {
                'state': TRACKING_SECOND_WAYPOINT,
                'tracking': 2
            }
        else:
            return state

    if state['state'] == TRACKING_SECOND_WAYPOINT:
        if reached_waypoint(2, gps_buffer, tolerance=WAYPOINT_TOLERANCE):
            rospy.loginfo('Begin tracking third waypoint')
            return {
                'state': TRACKING_THIRD_WAYPOINT,
                'tracking': 3
            }
        else:
            return state

    #NOTE AFTER WE FINISH CLIMBING DOWN, WE LOOP BACK TO TRACKING_THIRD_WAYPOINT
    if state['state'] == CLIMBING_UP:
        if going_down(gps_buffer,tolerance=RAMP_INCLINE_TOLERANCE):
            rospy.loginfo('Begin climbing down ramp')
            return {
                'state': CLIMBING_DOWN,
                'tracking': 3
            }
        else:
            return state

    if state['state'] == CLIMBING_DOWN:
        if going_flat(gps_buffer,tolerance=RAMP_INCLINE_TOLERANCE):
            rospy.loginfo('Begin following lines')
            return {
                'state': TRACKING_THIRD_WAYPOINT,
                'tracking': 3
            }
        else:
            return state

    if state['state'] == TRACKING_THIRD_WAYPOINT:
        if reached_waypoint(3, gps_buffer, tolerance=WAYPOINT_TOLERANCE):
            rospy.loginfo('Begin tracking second waypoint')
            return {
                'state': TRACKING_FOURTH_WAYPOINT,
                'tracking': 4
            }
        elif going_up(gps_buffer, tolerance=RAMP_INCLINE_TOLERANCE):
            rospy.loginfo('Begin climbing the ramp')
            return {
                'state': CLIMBING_UP,
                'tracking': 3
            }
        else:
            return state

    if state['state'] == TRACKING_FOURTH_WAYPOINT:
        if reached_waypoint(4, gps_buffer, tolerance=WAYPOINT_TOLERANCE):
            rospy.loginfo('Begin tracking second waypoint')
            return {
                'state': LINE_FOLLOWING,
                'tracking': 1
            }
        else:
            return state

    # if state['state'] == LINE_FOLLOWING:
    #
    #     # if we're within range of the first waypoint, start tracking it
    #     if reached_waypoint(0, gps_buffer, tolerance=FIRST_WAYPOINT_TOLERANCE):
    #         rospy.loginfo('Begin tracking first waypoint')
    #         return {
    #             'state': WAYPOINT_TRACKING,
    #             'speed': INITIAL_SPEED,
    #             'tracking': 0
    #         }
    #
    # if state['state'] == WAYPOINT_TRACKING:
    #     tracking = state['tracking']
    #
    #     # if we've reached the current waypoint, start tracking the next one
    #     if reached_waypoint(tracking, gps_buffer, tolerance=WAYPOINT_TOLERANCE):
    #
    #         # ... unless we are at the last one, in which case we should resume normal navigation
    #         if tracking == len(WAYPOINTS) - 1:
    #             rospy.loginfo('Reached all waypoints, resuming normal operation')
    #             return {
    #                 'state': LINE_FOLLOWING,
    #                 'speed': INITIAL_SPEED,
    #             }
    #
    #         next = tracking + 1
    #         rospy.loginfo('Begin tracking waypoint %s', next)
    #         return {
    #             'state': WAYPOINT_TRACKING,
    #             'speed': INITIAL_SPEED,
    #             'tracking': next
    #         }

    return state


scale = MAP_SIZE_METERS / float(MAP_SIZE_PIXELS)


def x_to_m(x):
    """converts x (pixel coordinate) to world coordinate"""
    return ((MAP_SIZE_PIXELS / -2.0) + x) * scale


def y_to_m(y):
    """converts y (pixel coordinate) to world coordinate"""
    return ((MAP_SIZE_PIXELS / 2.0) - y) * scale


def update_control((gps, costmap, pose, line_angle, state)):
    """figures out what we need to do based on the current state and map"""
    map_pose = costmap.transform

    transform = pose.transform.translation
    map_transform = map_pose.transform.translation
    diff = transform.x - map_transform.x, transform.y - map_transform.y
    diff = Vec2d.from_point(diff[0], diff[1])
    map_rotation = get_rotation(map_pose.transform)
    rotation = get_rotation(pose.transform)
    diff = diff.with_angle(diff.angle + map_rotation)
    diff_rotation = -rotation + map_rotation

    path = generate_path(costmap.costmap_bytes, diff_rotation, (diff.x, diff.y))

    if path is None:
        path = []
    path_debug.publish(
        Path(header=Header(frame_id='map'),
             poses=[PoseStamped(header=Header(frame_id='map'),
                                pose=Pose(position=Point(x=x_to_m(p[0]),
                                                         y=y_to_m(p[1]))))
                    for p in path]))
    print state
    # calculate theta_dot based on the current state
    if state['state'] == LINE_FOLLOWING or \
            state['state'] == TRACKING_THIRD_WAYPOINT:
        offset = 10
        if len(path) < offset + 1:
            goal = Vec2d(0, ATTRACTOR_THRESHOLD_MM)  # always drive forward
        else:
            point = path[offset]
            goal = Vec2d.from_point(x_to_m(point[0] + 0.5), y_to_m(point[1] + 0.5))
            goal = goal.with_magnitude(ATTRACTOR_THRESHOLD_MM)
        rotation = -line_angle.data  # rotate to follow lines
        if abs(rotation) < 10:
            rotation = 0

        rotation /= 1.0

    else:
        # FIXME
        goal = calculate_gps_heading(gps, WAYPOINTS[state['tracking'] - 1])  # track the waypoint

        rotation = to180(goal.angle)
        goal = goal.with_angle(0)  # don't need to crab for GPS waypoint, steering will handle that
        # state_debug.publish(str(goal))

    # calculate translation based on obstacles
    repulsors = extract_repulsors((diff.x, diff.y), costmap.map_bytes)
    potential = compute_potential(repulsors, goal)
    obstacle_debug.publish(PointCloud(header=Header(frame_id=topics.ODOMETRY_FRAME),
                                      points=[Point32(x=v.x / 1000.0, y=v.y / 1000.0) for v in repulsors]))
    translation = to180(potential.angle)

    # rospy.loginfo('translation = %s, rotation = %s, speed = %s', translation, rotation, INITIAL_SPEED)

    # don't rotate if bender needs to translate away from a line
    # if state['state'] == LINE_FOLLOWING:
    #     translation_threshhold = 60
    #     rotation_throttle = 0
    #     if np.absolute(translation) > translation_threshhold:
    #         rotation = rotation * rotation_throttle

    if state['state'] == CLIMBING_UP:
        speed = RAMP_SPEED
        rotation = gps['roll']
        rotation *= -10
        translation = 0
    elif state['state'] == CLIMBING_DOWN:
        speed = SLOW_SPEED
        rotation = gps['roll']
        rotation *= 10
        translation = 0
    else:
        obstacles = extract_repulsors((diff.x, diff.y), costmap.lidar_bytes)
        obstacles = [o for o in obstacles if abs(to180(o.angle)) < 45]
        min_dist = min(obstacles, key=lambda x: x.mag) if len(obstacles) > 0 else DIST_CUTOFF
        if min_dist < DIST_CUTOFF:
            speed = SLOW_SPEED
        else:
            speed = FAST_SPEED

    update_drivetrain(translation, rotation, speed)

    # rviz debug
    q = quaternion_from_euler(0, 0, math.radians(translation))
    heading_debug.publish(PoseStamped(header=Header(frame_id='map'),
                                      pose=Pose(position=Point(x=diff.x, y=diff.y),
                                                orientation=Quaternion(q[0], q[1], q[2], q[3]))))


def main():
    global control

    rospy.init_node(GUIDANCE_NODE)
    control = rospy.Publisher('input_vectors', Vector3, queue_size=3)

    gps = rx_subscribe(topics.GPS)

    # compute state based on GPS coordinates
    state = gps \
        .buffer_with_count(GPS_BUFFER, 1) \
        .scan(compute_next_state, seed=DEFAULT_STATE)

    # update controls whenever position or state emits
    tf = rx_subscribe('/tf', TFMessage, parse=None, buffer_size=100)

    costmap = rx_subscribe(topics.COSTMAP, String)

    line_angle = rx_subscribe(topics.LINE_ANGLE, Int16, parse=None).start_with(Int16(0))
    pos = tf.let(extract_tf(topics.ODOMETRY_FRAME)).start_with(
        TransformStamped(transform=Transform(rotation=Quaternion(0, 0, 0, 1))))

    pos.combine_latest(state, gps, costmap, line_angle,
                       lambda o, s, g, m, a: (g, m, o, a, s)) \
        .throttle_last(100) \
        .subscribe(on_next=update_control,
                   on_error=lambda e: rospy.logerr(traceback.format_exc(e)))

    rospy.spin()


if __name__ == '__main__':
    main()
