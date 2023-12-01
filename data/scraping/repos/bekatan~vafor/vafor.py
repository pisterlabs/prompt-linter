#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import sys, select
import tty, termios
import time
import spacy
import openai
import speech_recognition as sr
from darknet_ros_msgs.msg import BoundingBoxes
from std_msgs.msg import Float64MultiArray
from enum import Enum
from sensor_msgs.msg import LaserScan
from decouple import config

class Mode(Enum):
    IDLE = 0
    SEARCHING = 1
    FOUND = 2
    APPROACHING = 3
    REACHING = 4
    REACHING2 = 5
    GRIP = 6
    PICKUP = 7
    BRINGING = 8
    STARTUP = 9

BURGER_MAX_LIN_VEL = 0.22
BURGER_MAX_ANG_VEL = 2.84

WAFFLE_MAX_LIN_VEL = 0.26
WAFFLE_MAX_ANG_VEL = 1.82

LIN_VEL_STEP_SIZE = 0.01
ANG_VEL_STEP_SIZE = 0.1

TIME_TO_MOVE = 3

# Mode = Mode.IDLE

openai.api_key = config("OPENAI_API_KEY")
# objects = ['banana', 'bottle', 'remote']#, 'apple', 'chair', 'cup', 'keyboard', 'laptop', 'mouse', 'scissors', 'speaker', 'cellphone', 'chair']
objects = set()
# objects.add('apple')
# objects.add('none')
# objects.add('banana')
# objects.add('pizza')
# objects.add('chips')
# objects.add('cellphone')
# objects.add('bottle')
# objects.add('remote')
# objects.add('ibuprofen')
# objects.add('adrenaline')
# objects.add('epipen')
# objects.add('charger')
# objects.add('wrench')
# objects.add('screwdriver')
# objects.add('hammer')
# objects.add('tape measure')

msg = """
Welcome to the Vafor!
---------------------------
Press 'y' to make an inquiry
Press 'u' for home position
Press 'Space' to close grip

CTRL-C to quit
"""

e = """
Communications Failed
"""

def getKey():

    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def vels(target_linear_vel, target_angular_vel):
    return "currently:\tlinear vel %s\t angular vel %s " % (target_linear_vel,target_angular_vel)

def arms(angle1, angle2, angle3, angle4):
    tmp = "\n angle1 : %s \n angle2 : %s \n angle3 : %s \n angle4 : %s \n\n" % (angle1, angle2, angle3, angle4)
    return tmp

def makeSimpleProfile(output, input, slop):
    if input > output:
        output = min( input, output + slop )
    elif input < output:
        output = max( input, output - slop )
    else:
        output = input

    return output

def constrain(input, low, high):
    if input < low:
      input = low
    elif input > high:
      input = high
    else:
      input = input

    return input

def checkLinearLimitVelocity(vel):
    return constrain(vel, -WAFFLE_MAX_LIN_VEL, WAFFLE_MAX_LIN_VEL)
    

def checkAngularLimitVelocity(vel):
    return constrain(vel, -WAFFLE_MAX_ANG_VEL, WAFFLE_MAX_ANG_VEL)

#function to get the coco.names file
def get_coco_names():
    names = {}
    with open("coco.names", 'r') as f:
        for id, name in enumerate(f):
            names[id] = name[0:-1]
    return names

def getTarget(gpt = True):
    sentence = ""
    prev = ""
    target = ""
    while target == "":
        try:
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.5)
                
                audio2 = r.listen(source2)

                prev = sentence                
                sentence = r.recognize_google(audio2)
                sentence = sentence.capitalize()
                print("Did you say: \"",sentence,"\"?")
                
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
            
        except sr.UnknownValueError:
            print("unknown error occurred")
        
        if sentence == prev:
            continue

        if gpt:
            prompt = """
Which one object can help with the situation in the prompt the most?
Objects: {}.
Prompt: {}.
Answer:""".format(', '.join(objects), sentence)
            
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0,
                top_p = 0,
            )
            print(prompt)
            print(response.choices[0].text)

            doc = nlp(response.choices[0].text.strip())

            for token in doc:
                if token.lemma_.lower() == 'none':
                    print("Sorry I can not help you with that.")
                    break
                elif token.lemma_.lower() in objects:
                    target = token.lemma_.lower()
                    break
        else:
            doc = nlp(sentence)

            for token in doc:
                if token.dep_ == 'dobj' and token.text in names.values():
                    target = token.text
    
    return target

def detector(data):
    global Mode
    global sub_laser
    global sub_adjust
    if Mode == Mode.SEARCHING:
        for box in data.bounding_boxes:
            if box.Class == target:
                print("Found:", box.Class)
                sub_adjust = rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, adjust)
                sub_laser = rospy.Subscriber('/scan', LaserScan, laser)
                Mode = Mode.APPROACHING
                sub.unregister()
                return
    elif Mode == Mode.STARTUP:
        for box in data.bounding_boxes:
            objects.add(box.Class)

def laser(data):
    global Mode
    global direction
    
    if target == 'person' and data.ranges[0] < 0.7:
        Mode = Mode.REACHING2
        sub_laser.unregister()
        sub_adjust.unregister()
        return

    if data.ranges[0] < 0.3:
        Mode = Mode.REACHING
        sub_laser.unregister()
        sub_adjust.unregister()
        return

def adjust(data):
    global Mode
    global direction
    global target_angular_vel
    for bb in data.bounding_boxes:
        if bb.Class != target:
            continue
        center = (bb.xmin + bb.xmax) / 2
        if center > 635 and center < 645:
            target_angular_vel = 0
        elif center < 635:
            target_angular_vel = 0.02
        elif center > 645:
            target_angular_vel = -0.02

if __name__=="__main__":
    settings = termios.tcgetattr(sys.stdin)

    rospy.init_node('vafor')
    vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    arm_pub = rospy.Publisher('joint_trajectory_point', Float64MultiArray, queue_size=10)
    grib_pub = rospy.Publisher('gripper_position', Float64MultiArray, queue_size=10)

    turtlebot3_model = rospy.get_param("model", "burger")

    #initialize voice recognizer and language parser
    r = sr.Recognizer()
    nlp = spacy.load('en_core_web_lg')

    #initialize coco.names
    names = get_coco_names()
    target = ""

    direction = 0
    status = 0
    target_linear_vel   = 0.0
    target_angular_vel  = 0.0
    control_linear_vel  = 0.0
    control_angular_vel = 0.0

    default_angles = 0.0, -1.2, 0.78, 0.51
    reach_angles = 0.0, 1.08, -0.33, -0.2
    reach_angles2 = 0.0, 1.08, -0.6, -0.42
    holdup_angles = 0.0, 0.0, -1.6, 0.0
    angle1, angle2, angle3, angle4 = default_angles
    
    Mode = Mode.STARTUP
    prev = Mode.STARTUP
    grib_angle = 0.025
    sub = rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, detector)
    start = time.time()
    try:
        print(msg)
        while not rospy.is_shutdown():
            if Mode != prev:
                print("Mode changed from: ", prev, " to: ", Mode)
                prev = Mode
                if Mode == Mode.IDLE:
                    print(msg)
                    print("The objects: ", objects)
            key = getKey()
            if Mode == Mode.STARTUP:
                target_angular_vel = -ANG_VEL_STEP_SIZE
                target_linear_vel   = 0.0
                if time.time() - start > 10:
                    sub.unregister()
                    Mode = Mode.IDLE
            if Mode == Mode.IDLE:
                target_linear_vel   = 0.0
                control_linear_vel = 0.0
                target_angular_vel  = 0.0
                control_angular_vel = 0.0
                grib_angle = 0.025
                angle1, angle2, angle3, angle4 = default_angles
            elif Mode == Mode.SEARCHING:
                target_angular_vel = ANG_VEL_STEP_SIZE
                target_linear_vel   = 0.0
                # angle1, angle2, angle3, angle4 = default_angles
            elif Mode == Mode.APPROACHING:
                target_linear_vel   = LIN_VEL_STEP_SIZE
            elif Mode == Mode.REACHING:
                target_linear_vel   = 0.0
                control_linear_vel = 0.0
                target_angular_vel  = 0.0
                control_angular_vel = 0.0

                angle1, angle2, angle3, angle4 = reach_angles
                Mode = Mode.REACHING2
            elif Mode == Mode.REACHING2:
                time.sleep(TIME_TO_MOVE)

                target_linear_vel   = 0.0
                control_linear_vel = 0.0
                target_angular_vel  = 0.0
                control_angular_vel = 0.0

                angle1, angle2, angle3, angle4 = reach_angles2
                Mode = Mode.GRIP
            elif Mode == Mode.GRIP:
                time.sleep(TIME_TO_MOVE)
                if target != 'person':
                    grib_angle = -0.1
                    Mode = Mode.PICKUP
                else:
                    grib_angle = 0.025
                    Mode = Mode.IDLE
            elif Mode == Mode.PICKUP:
                time.sleep(TIME_TO_MOVE)
                angle1, angle2, angle3, angle4 = default_angles
                Mode = Mode.BRINGING
            elif Mode == Mode.BRINGING:
                target = 'person'
                Mode = Mode.SEARCHING
                sub = rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, detector)

            if key == 'y':
                print("How can I help you?")
                target = getTarget()
                print("Target is \"",target,"\"")
                Mode = Mode.SEARCHING
                sub = rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, detector)
                
            elif key == 'u' :
                print("Stoppping")
                angle1, angle2, angle3, angle4 = default_angles
                Mode = Mode.IDLE
            
            elif key == 'r' :
                print("Reaching")
                angle1, angle2, angle3, angle4 = reach_angles
                print(arms(angle1, angle2, angle3, angle4))

            elif key == 'w' :
                target_linear_vel = checkLinearLimitVelocity(target_linear_vel + LIN_VEL_STEP_SIZE)
                status = status + 1
                print(vels(target_linear_vel,target_angular_vel))
            elif key == 'x' :
                target_linear_vel = checkLinearLimitVelocity(target_linear_vel - LIN_VEL_STEP_SIZE)
                status = status + 1
                print(vels(target_linear_vel,target_angular_vel))
            elif key == 'a' :
                target_angular_vel = checkAngularLimitVelocity(target_angular_vel + ANG_VEL_STEP_SIZE)
                status = status + 1
                print(vels(target_linear_vel,target_angular_vel))
            elif key == 'd' :
                target_angular_vel = checkAngularLimitVelocity(target_angular_vel - ANG_VEL_STEP_SIZE)
                status = status + 1
                print(vels(target_linear_vel,target_angular_vel))
            elif key == 's' :
                target_linear_vel   = 0.0
                control_linear_vel  = 0.0
                target_angular_vel  = 0.0
                control_angular_vel = 0.0
                angle1, angle2, angle3, angle4 = default_angles
                print(vels(target_linear_vel, target_angular_vel))
            elif key == '0':
                angle1 += 0.03
                print(arms(angle1, angle2, angle3, angle4))
            elif key == '1':
                angle2 += 0.03
                print(arms(angle1, angle2, angle3, angle4))
            elif key == '4':
                angle3 += 0.03
                print(arms(angle1, angle2, angle3, angle4))
            elif key == '7':
                angle4 += 0.03
                print(arms(angle1, angle2, angle3, angle4))
            elif key == '.':
                angle1 -= 0.03
                print(arms(angle1, angle2, angle3, angle4))
            elif key == '2':
                angle2 -= 0.03
                print(arms(angle1, angle2, angle3, angle4))
            elif key == '5':
                angle3 -= 0.03
                print(arms(angle1, angle2, angle3, angle4))
            elif key == '8':
                angle4 -= 0.03
                print(arms(angle1, angle2, angle3, angle4))
            elif key == ' ':
                if grib_angle == 0.025:
                    grib_angle = -0.1
                else:
                    grib_angle = 0.025
                
            else:
                if (key == '\x03'):
                    break

            if status == 20 :
                print(msg)
                status = 0

            twist = Twist()

            control_linear_vel = makeSimpleProfile(control_linear_vel, target_linear_vel, (LIN_VEL_STEP_SIZE/2.0))
            twist.linear.x = control_linear_vel; twist.linear.y = 0.0; twist.linear.z = 0.0

            control_angular_vel = makeSimpleProfile(control_angular_vel, target_angular_vel, (ANG_VEL_STEP_SIZE/2.0))
            twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = control_angular_vel

            vel_pub.publish(twist)

            arm_msg = Float64MultiArray()
            arm_msg.data = [-1, angle1, angle2, angle3, angle4]
            arm_pub.publish(arm_msg)
            
            grib_msg = Float64MultiArray()
            grib_msg.data = [grib_angle]
            grib_pub.publish(grib_msg)


            

    except:
        print("Some error:", e)

    finally:
        twist = Twist()
        twist.linear.x = 0.0; twist.linear.y = 0.0; twist.linear.z = 0.0
        twist.angular.x = 0.0; twist.angular.y = 0.0; twist.angular.z = 0.0
        vel_pub.publish(twist)

        arm_msg = Float64MultiArray()
        arm_msg.data = [-1, 0.0, -1.2, 0.78, 0.51]
        arm_pub.publish(arm_msg)

        grib_msg = Float64MultiArray()
        grib_msg.data = [0.025]
        grib_pub.publish(grib_msg)


    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
