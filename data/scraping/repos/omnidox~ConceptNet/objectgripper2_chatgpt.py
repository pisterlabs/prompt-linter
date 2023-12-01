import rospy
import tf
import moveit_commander
import geometry_msgs.msg
from geometry_msgs.msg import Pose, Point, Quaternion, PoseArray
from std_msgs.msg import String, Float64MultiArray
from visualization_msgs.msg import Marker
import numpy as np
from sensor_msgs.msg import JointState
import sys
import threading
import pandas as pd
import json
import openai
import configparser




child_frame = []
fifth_joint = 0
list_filled_event = threading.Event()

def get_gpt_context(object_name, desired_contexts=[], tasks=[]):

    # Start timing for the first implementation
    start_time_1 = time.time()


    # Set up the API key
    config = configparser.ConfigParser()
    config.read('config.ini')
    openai.api_key = config['DEFAULT']['OPENAI_API_KEY']

   # If no specific contexts are provided, consider all available contexts
    if not desired_contexts:
        desired_contexts = ["kitchen", "office", "child's_bedroom", "living_room", "bedroom", 
                            "dining_room", "pantry", "garden", "laundry_room"]

    # Refine the prompt to guide the model towards a shorter answer
    prompt = f"Which of the following contexts is the object '{object_name}' most likely associated with: {', '.join(desired_contexts)}? Please specify only the context as a response."



    # Determine the system message based on tasks
    if tasks:
        task_str = " and ".join(tasks)
        messages = [
            {"role": "system", "content": f"You are a helpful {task_str} assistant robot."},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant robot."},
            {"role": "user", "content": prompt}
        ]

    for message in messages:
        print(f"{message['role']}: {message['content']}")

    # Use the chat interface
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.6,  # Adjusts the randomness of the output
        top_p=0.9,  # Adjusts the nucleus sampling
        messages=messages
    )

    # Extract the model's response
    answer = response['choices'][0]['message']['content'].strip()

    # Post-process the response to extract just the context
    # This step can be refined further based on the model's typical responses
    for context in desired_contexts:
        if context in answer:
            end_time_1 = time.time()

            execution_time_1 = end_time_1 - start_time_1
            print(f"Execution time for the chatgpt implementation: {execution_time_1:.4f} seconds")

            return context
        
    end_time_1 = time.time()

    execution_time_1 = end_time_1 - start_time_1
    print(f"Execution time for the chatgpt implementation: {execution_time_1:.4f} seconds")

    return answer  # Return the raw answer if no context is found

def callback(msg):
    # Define the callback function to handle the incoming messages
    counter = 0
    while counter < 4:
        if msg.text not in child_frame:
            child_frame.append(msg.text)
        counter += 1
    list_filled_event.set()
    #print(child_frame)

def joint_callback(msg):
    global fifth_joint
    fifth_joint = msg.position[4]
    #print(fifth_joint)
def move_forward(rot, trans, z):

    move_group.set_start_state_to_current_state()
    start_pose = move_group.get_current_pose().pose
    obj_pose = geometry_msgs.msg.Pose()
    obj_pose.position.x = (trans[0])
    obj_pose.position.y = (trans[1])
    obj_pose.position.z = z
    obj_pose.orientation.x = rot[0]
    obj_pose.orientation.y = rot[1]
    obj_pose.orientation.z = rot[2]
    obj_pose.orientation.w = rot[3]

    print(obj_pose.position.x)

    waypoints = [start_pose, obj_pose]
    end_effector_point = waypoints[-1]
    prev_point = waypoints[-2]
    end_effector_pose = Pose()
    end_effector_pose.position.x = (end_effector_point.position.x*.75)
    end_effector_pose.position.y = end_effector_point.position.y
    end_effector_pose.position.z = z
    prev_point_orientation = prev_point.orientation
    end_effector_pose.orientation.x = prev_point_orientation.x
    end_effector_pose.orientation.y = prev_point_orientation.y
    end_effector_pose.orientation.z = prev_point_orientation.z
    end_effector_pose.orientation.w = prev_point_orientation.w
    waypoints[-1] = end_effector_pose

    move_group.set_pose_target(waypoints[-1])
    plan = move_group.plan()
    plan_traj = plan[1]
    move_group.execute(plan_traj, wait=True)
    move_group.stop()
    move_group.clear_pose_targets()
    rospy.sleep(6)

def move_up(rot, trans):
    move_group.set_start_state_to_current_state()
    start_pose = move_group.get_current_pose().pose
    obj_pose = geometry_msgs.msg.Pose()
    obj_pose.position.x = (trans[0])
    obj_pose.position.y = (trans[1])
    obj_pose.position.z = (trans[2] + .45)
    obj_pose.orientation.x = rot[0]
    obj_pose.orientation.y = rot[1]
    obj_pose.orientation.z = rot[2]
    obj_pose.orientation.w = rot[3]

    print(obj_pose.position.x)

    waypoints = [start_pose, obj_pose]
    end_effector_point = waypoints[-1]
    prev_point = waypoints[-2]
    end_effector_pose = Pose()
    end_effector_pose.position.x = (end_effector_point.position.x * .70)
    end_effector_pose.position.y = end_effector_point.position.y
    end_effector_pose.position.z = (end_effector_point.position.z)
    prev_point_orientation = prev_point.orientation
    end_effector_pose.orientation.x = prev_point_orientation.x
    end_effector_pose.orientation.y = prev_point_orientation.y
    end_effector_pose.orientation.z = prev_point_orientation.z
    end_effector_pose.orientation.w = prev_point_orientation.w
    waypoints[-1] = end_effector_pose

    move_group.set_pose_target(waypoints[-1])
    plan = move_group.plan()
    plan_traj = plan[1]
    move_group.execute(plan_traj, wait=True)
    move_group.stop()
    move_group.clear_pose_targets()
    rospy.sleep(10)

def move_down2(rot, trans):
    move_group.set_start_state_to_current_state()
    start_pose = move_group.get_current_pose().pose
    obj_pose = geometry_msgs.msg.Pose()
    obj_pose.position.x = (trans[0])
    obj_pose.position.y = (trans[1])
    obj_pose.position.z = (trans[2]+ .038) #might need to be adjusted for certain items
    obj_pose.orientation.x = rot[0]
    obj_pose.orientation.y = rot[1]
    obj_pose.orientation.z = rot[2]
    obj_pose.orientation.w = rot[3]
    print("before the ifs, fifth joint: ", fifth_joint)
    print(obj_pose.position.z)
    if fifth_joint > .03:
        print("fifth joint right: ", fifth_joint)
        waypoints = [start_pose, obj_pose]
        end_effector_point = waypoints[-1]
        prev_point = waypoints[-2]
        end_effector_pose = Pose()
        end_effector_pose.position.x = (end_effector_point.position.x - .05)#.05
        end_effector_pose.position.y = (end_effector_point.position.y + .016)#.012
        end_effector_pose.position.z = (end_effector_point.position.z)
        prev_point_orientation = prev_point.orientation
        end_effector_pose.orientation.x = prev_point_orientation.x
        end_effector_pose.orientation.y = prev_point_orientation.y
        end_effector_pose.orientation.z = prev_point_orientation.z
        end_effector_pose.orientation.w = prev_point_orientation.w
        waypoints[-1] = end_effector_pose

        move_group.set_pose_target(waypoints[-1])
        plan = move_group.plan()
        plan_traj = plan[1]
        move_group.execute(plan_traj, wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        rospy.sleep(5)
    elif fifth_joint < -.03:
        print("fifth joint left: ", fifth_joint)
        waypoints = [start_pose, obj_pose]
        end_effector_point = waypoints[-1]
        prev_point = waypoints[-2]
        end_effector_pose = Pose()
        end_effector_pose.position.x = (end_effector_point.position.x - .05)
        end_effector_pose.position.y = (end_effector_point.position.y + .05)
        end_effector_pose.position.z = (end_effector_point.position.z)
        prev_point_orientation = prev_point.orientation
        end_effector_pose.orientation.x = prev_point_orientation.x
        end_effector_pose.orientation.y = prev_point_orientation.y
        end_effector_pose.orientation.z = prev_point_orientation.z
        end_effector_pose.orientation.w = prev_point_orientation.w
        waypoints[-1] = end_effector_pose

        move_group.set_pose_target(waypoints[-1])
        plan = move_group.plan()
        plan_traj = plan[1]
        move_group.execute(plan_traj, wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        rospy.sleep(5)
    else:
        waypoints = [start_pose, obj_pose]
        end_effector_point = waypoints[-1]
        prev_point = waypoints[-2]
        end_effector_pose = Pose()
        end_effector_pose.position.x = (end_effector_point.position.x - .05)
        end_effector_pose.position.y = (end_effector_point.position.y + .027)
        end_effector_pose.position.z = (end_effector_point.position.z)
        prev_point_orientation = prev_point.orientation
        end_effector_pose.orientation.x = prev_point_orientation.x
        end_effector_pose.orientation.y = prev_point_orientation.y
        end_effector_pose.orientation.z = prev_point_orientation.z
        end_effector_pose.orientation.w = prev_point_orientation.w
        waypoints[-1] = end_effector_pose

        move_group.set_pose_target(waypoints[-1])
        plan = move_group.plan()
        plan_traj = plan[1]
        move_group.execute(plan_traj, wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        rospy.sleep(5)

if __name__ == "__main__":
    rospy.init_node('Rafael_panda_move')
    rospy.Subscriber('/object', Marker, callback)
    rospy.Subscriber('/joint_states', JointState, joint_callback)
    moveit_commander.roscpp_initialize(sys.argv)
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_name = "panda_arm"
    move_group = moveit_commander.MoveGroupCommander(group_name)


    move_group.set_planning_time(10)
    move_group.set_end_effector_link("panda_hand")
    # Get the initial pose of the end effector
    start_pose = move_group.get_current_pose().pose

    # Initialize the transform listener
    listener = tf.TransformListener()

    # Initialize the gripper commander
    group_hand_name = "hand"
    gripper = moveit_commander.MoveGroupCommander(group_hand_name)

    # Initialize the gripper commander

    # Define the joint state to close the gripper
    gripper_joint_goal = [0.35, 0.35] 
    joint_home = [-0.0002249274015897828, -0.7846473935524318, 4.74312715691906e-06, -2.3559378868236878, -0.0007421279900396864, 1.571840799410771, 0.78462253368357]
    joints = []
    list_filled_event.wait()


    while not rospy.is_shutdown():
        for object in child_frame[:]:
            
            print("object", object)
            print("child frames", child_frame)
            i = 0
            move_group.set_start_state_to_current_state()
            start_pose = move_group.get_current_pose().pose

            listener.waitForTransform('panda_link0', 'panda_hand', rospy.Time(), rospy.Duration(3.0))

            (trans2, rot2) = listener.lookupTransform('panda_link0', 'panda_hand', rospy.Time(0))

            arm_pose = geometry_msgs.msg.Pose()
            arm_pose.position.x = trans2[0]
            arm_pose.position.y = trans2[1]
            arm_pose.position.z = (trans2[2])
            
            listener.clear()


            move_group.set_start_state_to_current_state()
            start_pose = move_group.get_current_pose().pose

            listener.waitForTransform('panda_link0', object,rospy.Time(), rospy.Duration(5.0))

            (trans, rot) = listener.lookupTransform('panda_link0', object, rospy.Time(0))
            print(f"rotation: {rot}, translation: {trans}, object: {object}")
            

            move_forward(rot, trans, arm_pose.position.z)

            listener.clear()

            move_group.set_start_state_to_current_state()
            start_pose = move_group.get_current_pose().pose
            

            listener.waitForTransform('panda_link0', object,rospy.Time(), rospy.Duration(5.0))

            (trans, rot) = listener.lookupTransform('panda_link0', object, rospy.Time(0))
            print(f"rotation: {rot}, translation: {trans}, object: {object}")


            move_up(rot, trans)

            listener.clear()
            move_group.set_start_state_to_current_state()
            start_pose = move_group.get_current_pose().pose


            listener.waitForTransform('panda_link0', object,rospy.Time(), rospy.Duration(5.0))

            (trans, rot) = listener.lookupTransform('panda_link0', object, rospy.Time(0))
            print(f"rotation: {rot}, translation: {trans}, object: {object}")


            move_down2(rot, trans)
            
            hand_close = [0.02, 0.02]
            gripper.set_joint_value_target([0.02, 0.02])
            plan = gripper.plan()
            plan_traj = plan[1]
            gripper.execute(plan_traj, wait=True)
            gripper.stop()
            rospy.sleep(1)
            


            contexts_csv = '/home/parronj1/Rafael/local_detectron/contexts.csv'
            df = pd.read_csv(contexts_csv)
            desired_contexts = []
            desired_contexts.extend(df["Context"])

            # Prompt the user for tasks
            available_contexts = ["kitchen", "office", "child's_bedroom", "living_room", "bedroom", 
                                "dining_room", "pantry", "garden", "laundry_room"]

            prompt_message = ("Please input what contexts or multiple contexts separated by commas for the robot to focus on. "
                            f"These are the possible contexts: {', '.join(available_contexts)} "
                            "(or press Enter to continue without specifying tasks): ")

            while True:
                user_input = input(prompt_message).strip()

                # Split the input by commas and strip whitespace
                tasks = [task.strip() for task in user_input.split(",")] if user_input else []

                # Check if all tasks are in available_contexts
                if all(task in available_contexts for task in tasks) or not tasks:
                    if tasks:
                        print(f"You have set the robot's focus on: {', '.join(tasks)}")
                    else:
                        print("There will be no focus.")
                    break

                else:
                    print("One or more of the contexts you entered are not valid. Please try again.")

            context = get_gpt_context(object, desired_contexts = desired_contexts,tasks=tasks )
            print(f"The most relevant context for {object} is {context}.")

            
            context = context.strip().lower()
            filtered_rows = df[df["Context"] == context]
            

            if not filtered_rows.empty:
                matching_row = filtered_rows.iloc[0]
                row_values = matching_row.values[1:]
                print(row_values)
                joints.extend(row_values)

            

            print(joints)
            move_group.set_joint_value_target(joint_home)
            plan = move_group.plan()
            plan_traj = plan[1]
            move_group.execute(plan_traj, wait=True)
            move_group.stop()
            move_group.clear_pose_targets()


            move_group.set_joint_value_target(joints)
            plan = move_group.plan()
            plan_traj = plan[1]
            move_group.execute(plan_traj, wait=True)
            move_group.stop()
            move_group.clear_pose_targets()
            rospy.sleep(2)

            joints.clear()

            

            hand_open = [0.036, 0.036]
            gripper.set_joint_value_target([0.036, 0.036])
            plan = gripper.plan()
            plan_traj = plan[1]
            gripper.execute(plan_traj, wait=True)
            gripper.stop()
            rospy.sleep(2)

            move_group.set_joint_value_target(joint_home)
            plan = move_group.plan()
            plan_traj = plan[1]
            move_group.execute(plan_traj, wait=True)
            move_group.stop()
            move_group.clear_pose_targets()
            rospy.sleep(8)

            print("before pop", child_frame)

            child_frame.remove(object)
            listener.clear()
            print("after pop", child_frame)
            print("object after pop", object)
        if len(child_frame) == 0:
            print("nothing in child frame")
            exit()
            
    rospy.spin()

      
