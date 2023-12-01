import os
import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile

from rclpy.qos import qos_profile_sensor_data

from rclpy.executors import Executor
from rclpy.executors import MultiThreadedExecutor
from rclpy.executors import SingleThreadedExecutor


from std_msgs.msg import String
from geometry_msgs.msg import Pose

from custom_interfaces.srv import BimanualJson
from custom_interfaces.srv import UserInput


from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
)
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import SystemMessage

from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI

import time
import json

from typing import Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# yes i know its looks bad with the ros2 stuff  

#global var to store robot state 
robot_full_pose = {"step_1":{   "left_ee_coor"           : [1.1145,-0.2700,1.1762],
                                "right_ee_coor"          : [-1.1145,-0.2700,1.1762],
                                "left_ee_orientation"    : [0.27,0.27,0.653,0.653],
                                "right_ee_orientation"   : [0,0,0,1],
                                "left_gripper_state"     : False,
                                "right_gripper_state"    : False}}

#global var to store the item dict
item_dict_global = None

def send_pose_to_robots(pose: dict):
    pose_str = json.dumps(pose)
    planner_client = PlannerClient()
    planner_client.get_logger().info(f"sending pose str: {pose_str}")
    result = planner_client.send_request(pose_str)
    planner_client.destroy_node()
    return result

#------------------------functions-to-work-as-llm-tools----------------------

def move_single_arm(side:str, coordinates, orientation):
    if side == 'right':
        robot_full_pose["step_1"]["right_ee_coor"] = coordinates
        robot_full_pose["step_1"]["right_ee_orientation"] = orientation
    elif side == 'left':
        robot_full_pose["step_1"]["left_ee_coor"] = coordinates
        robot_full_pose["step_1"]["left_ee_orientation"] = orientation
    result = send_pose_to_robots(robot_full_pose)
    return [result.success, result.msg]



def move_both_arms(left_coordinates, left_orientation, right_coordinates, right_orientation):
    robot_full_pose["step_1"]["right_ee_coor"] = right_coordinates
    robot_full_pose["step_1"]["right_ee_orientation"] = right_orientation
    robot_full_pose["step_1"]["left_ee_coor"] = left_coordinates
    robot_full_pose["step_1"]["left_ee_orientation"] = left_orientation
    result = send_pose_to_robots(robot_full_pose)
    return [result.success, result.msg]

def use_gripper(side, state):
    if side == 'right':
        robot_full_pose["step_1"]["right_gripper_state"] = state
    if side == 'left':
        robot_full_pose["step_1"]["left_gripper_state"] = state
    result = send_pose_to_robots(robot_full_pose)
    return [result.success, result.msg]

def get_item_dict(empty = None):
    return item_dict_global


def get_full_robot_pose(empty = None):
    return robot_full_pose

def get_pre_grasp_pose(object_position):
    op = object_position
    pose = {'end_effector_position': [op[0], op[1], op[2]+0.2],
            'end_effector_orientation':[0,0,0,1]} #fix this later
    return pose

def get_grasp_pose(object_position):
    op = object_position
    pose = {'end_effector_position' : op,
            'end_effector_orientation': [0,0,0,1]} #fix this later 
    return pose 


def grasp_object(side, object_position):
    pre_grasp_pose = get_pre_grasp_pose(object_position)
    move_pre_grasp_result = move_single_arm(side,pre_grasp_pose['end_effector_position'],pre_grasp_pose["end_effector_orientation"])
    grasp_pose = get_grasp_pose(object_position)
    move_grasp_result = move_single_arm(side,grasp_pose['end_effector_position'],grasp_pose["end_effector_orientation"])
    use_gripper_result = use_gripper(side,True)
    return {'move_pre_grasp_result':move_pre_grasp_result,
            'move_grasp_result':move_grasp_result,
            'use_gripper_result':use_gripper_result}

class MoveSingleArmInput(BaseModel):
    """Inputs for move_single_arm"""
    side: str = Field(description="Which robot arm to use, the left or the right", examples=['left','right'])
    coordinates: list = Field(description="The coordinates of the end effector of the robot in meters", examples=[[0.423,0.123,0.234],[-0,324,0.533,0.543]])
    orientation: list = Field(description="The rotation of the end effector on the manipulator represented in quaturnions [x, y, z, w]", examples=[[0,0,0,1]]) #prehabs add more examples 

class MoveSingleArmTool(BaseTool):
    name = "move_single_arm"
    description = """
        Usefull for when you want to move a single arms end effector to the a set of coordinates and a given orientation
        Outputs a list where the first value is whether or not the move was a success and the second value is the error message"""
    args_schema: Type[BaseModel] = MoveSingleArmInput
    def _run(self, side, coordinates, orientation):
        result = move_single_arm(side, coordinates, orientation)
        return result

class MoveBothArmsInput(BaseModel):
    """Inputs for move_both_arms"""
    left_coordinates: list = Field(description="The coordinates of the left arm end effector in [x ,y, z] order. Must be floats", examples=[[-0.423,0.123,0.234],[-0,324,0.533,0.543]])
    left_orientation: list =  Field(description="The rotation of the left end effector on the manipulator represented in quaturnions [x, y, z, w]", examples=[[0,0,0,1]]) #prehabs add more examples 
    right_coordinates: list = Field(description="The coordinates of the right arm end effector in [x ,y, z] order. Must be floats", examples=[[0.423,0.123,0.234],[0,324,0.533,0.543]])
    right_orientation: list = Field(description="The rotation of the right end effector on the manipulator represented in quaturnions [x, y, z, w]", examples=[[0,0,0,1]]) #prehabs add more examples 

class MoveBothArmsTool(BaseTool):
    name = "move_both_arms"
    description = """
        Usefull for when you want to move both arms to each their coordinates and end effector orientation. Note that both arms must not be at the same position.
        Outputs a list where the first value is whether or not the move was a success and the second value is the error message"""
    args_schema: Type[BaseModel] = MoveBothArmsInput
    def _run(self,left_coordinates, left_orientation, right_coordinates, right_orientation):
        result = move_both_arms(left_coordinates, left_orientation, right_coordinates, right_orientation)
        return result
    
class UseGripperInput(BaseModel):
    """Inputs for use_gripper"""
    side:str = Field(description="Which arm the gripper is mounted on that you want to use", examples=['left', 'right'])
    state:bool = Field(description="If the gripper should be open or closed eg.: True = close, False = open", examples=[True,False])

class UseGripperTool(BaseTool):
    name = "use_gripper"
    description = """
        Usefull for when you want to open or close a gripper.
        Outputs a list where the first value is whether or not the move was a success and the second value is the error message"""
    args_schema: Type[BaseModel] = UseGripperInput
    def _run(self,side,state):
        result = use_gripper(side,state)
        return result

class GetItemDictInput(BaseModel):
    """input for get_item_dict"""
    empty:bool = Field(description="empty value")

class GetItemDictTool(BaseTool):
    name = "get_item_dict"
    description = """
        Usefull for when you want to know what items are pressent in your workspace and where these items are located.
        Outputs a dictionary of the objects presssent and their location"""
    args_schema: Type[BaseModel] = GetItemDictInput
    def _run(self, empty):
        result = get_item_dict(empty)
        return result

class GetFullRobotPoseInput(BaseModel):
    """input for get_full_robot_pose"""
    empty:bool = Field(description="empty value")

class GetFullRobotPoseTool(BaseTool):
    name = "get_full_robot_pose"
    description = """
        Usefull for when you want to know the current pose of the robot 
        Outputs a dictionary containing: the left arm's end effector position in the global coordinate system ['left_ee_coor'], 
        the right arm's end effector position in the global coordinate system ['right_ee_coor'], 
        the left arm's end effector orientation in x, y, z, w quaternions ['left_ee_orientation'], 
        the right arm's end effector orientation in quaternions ['right_ee_orientation'], 
        the left arm's gripper state where False = open anf True = closed ['left_gripper_state'], and the right arm's gripper state ['right_gripper_state'] """
    args_schema: Type[BaseModel] = GetFullRobotPoseInput
    def _run(self,empty):
        result = get_full_robot_pose(empty)
        return result
    
class GetPreGraspPoseInput(BaseModel):
    """input for get_pre_grasp_pose"""
    object_position:list[float] = Field(description="The global position on the object that you want to grasp")

class GetPreGraspPoseTool(BaseTool):
    name = "get_pre_grasp_pose"
    description = """
        Usefull for when you want to find the end effector pose before going to the grasp postition an object.
        Always use this function before grasping an object by moving the arm that you want to use for grasping to the pose outputted by this function to ensure the correct aproach.
        Outputs a dictionary containing the end effecor position ['end_effector_positiom'] and end effector orientation ['end_effector_orientation']
        The end effector position is in the global frame
        Make sure to always use the get_grasp_pose and go to that position after using this tool before grasping the object"""
    args_schema: Type[BaseModel] = GetPreGraspPoseInput
    def _run(self, object_position):
        pre_grasp_pose = get_pre_grasp_pose(object_position)
        return pre_grasp_pose

class GetGraspPoseInput(BaseModel):
    """input for get_grasp_pose"""
    object_position:list[float] = Field(description="the global position of the object that you want to grasp")

class GetGraspPoseTool(BaseTool):
    name = "get_grasp_pose"
    description = """
        Usefull for when you want to find the pose of the end effector to grasp an object 
        Use this after you have moved the arm to the pre grasp pose. Then move the robot to the pose outputted by this tool
        Outputs a dictionary containing the end effecor position ['end_effector_positiom'] and end effector orientation ['end_effector_orientation']
        The end effector position is in the global frame"""
    args_schema: Type[BaseModel] = GetGraspPoseInput
    def _run(self, object_position):
        grasp_pose = get_grasp_pose(object_position)
        return grasp_pose

class GraspObjectInput(BaseModel):
    """input to grasp_object"""
    side:str = Field(description="The robot arm that you want to use to grasp the object",examples=['left','right'])
    object_position: list[float] = Field(description="the global position of the object that you want to grasp")

class GraspObjectTool(BaseTool):
    name = "grasp_object"
    description = """
        Usefull for when you want to grasp an object using a single robot arm. 
        It starts bpy moving the robot to the pre grasp position then to the gras postion and then closes the gripper around the object in the given position.
        Outputs a dictionary containting data from each step"""
    args_schema: Type[BaseModel] = GraspObjectInput
    def _run(self,side,object_position):
        return grasp_object(side=side,object_position=object_position)
#------------------------end of llm tools-----------------------------





#-----------------------robot planner comunication node---------------
class PlannerClient(Node):
    def __init__(self):
        super().__init__('planner_client')
        self.cli = self.create_client(BimanualJson, 'llm_executor')
        
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
            

    def send_request(self, task):
        #creating empty request 
        self.req = BimanualJson.Request()
        self.req.json_steps = task
        
        self.future = self.cli.call_async(self.req)

        self.get_logger().info(f"node exec:{self.executor}, future exec:{self.future._executor}")

        rclpy.spin_until_future_complete(node=self, future=self.future,)
        
        self.get_logger().info("result recived")
        return self.future.result()

#-----------------------the gpt controller node----------------------
class GptController(Node):

    def __init__(self):
        # Here you have the class constructor
        # call the class constructor
        super().__init__('gpt_controlle')
        
        #-------------------communication with other nodes------------------------
        self.srv = self.create_service(UserInput, 
                                       'user_input_srv',
                                       self.user_input_callback) 

        
        self._item_dict_sub  = self.create_subscription(String, 
                                                        '/item_dict', 
                                                        self.item_dict_callback,qos_profile_sensor_data)

        
        self._right_robot_position_sub= self.create_subscription(Pose, 
                                                                 '/state_pose_right', 
                                                                 self.right_robot_position_callback,
                                                                 qos_profile_sensor_data)
        
        self._left_robot_position_sub = self.create_subscription(Pose, 
                                                                 '/state_pose_left', 
                                                                 self.left_robot_position_callback,
                                                                 qos_profile_sensor_data)
        
        # item dict value if nothing is recieved 
        self.item_dict =False 
        
        #---------------langchain and openai setup---------------------

        #creating the system message for the agent llm 
        system_message = SystemMessage(content="You are controlling a bimanual robot. Use the tools provided to sovle the users problem. To solve the users problem start by breaking the task into smaller steps that you can solve using a single tool call for each step")

        #defining the model to ofe with the llm
        llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

        #defining the list of tools availble for the agent llm to use 
        tools = [MoveSingleArmTool(), 
                 MoveBothArmsTool(), 
                 UseGripperTool(), 
                 GetItemDictTool(), 
                 GetFullRobotPoseTool(), 
                 #GetPreGraspPoseTool(), 
                 #GetGraspPoseTool(),
                 GraspObjectTool()]
        #finaly defining the agent llm 
        agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True ,system_message=system_message)
        
        
        #-----------------------------setup-planner-------------------------
        
        #creating a string containing information about the tools used by the agent 
        tools_dict = {}

        for tool in tools:
            t = tool
            tools_dict[t.name] = t.description

        agent.from_agent_and_tools

        self.tools_str = json.dumps(tools_dict)

        #creating the prompt templates 
        system_message_planner = """You are a helpful assistant that creates a detailed plan for a robot to follow to solve the task given by the user.
        To solve the task, tools from this list can be used:
        {function_list}
        Return only the plan as a step-by-step manual for the robot operator to follow.
        Do not return anything but the steps to follow"""


        user_template = """{task}"""

        #defining the prompt template 
        self.chat_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_message_planner),
                ("human", user_template),
            ]
        )

        #defining the modle to use for the planner llm 
        model = ChatOpenAI(model_name="gpt-4-1106-preview")

        #---------------------
        self.llm_outputs = {'planner':None, 'agent':None}

        

        #-----------------------connected-chain------------------------ 
        #this the the langchain llm chain staring with the planner llm which is parsed to the agent llm 
        self.chain = model | StrOutputParser() | self.save_llm_output  | agent.run | StrOutputParser() | self.save_llm_output
    
    
    
    def save_llm_output(self, llm_output):
        if self.llm_outputs['planner'] == None:
            self.llm_outputs['planner'] = llm_output
            
        else:
            self.llm_outputs['agent'] = llm_output  
        
        with open('llm_output.json','w') as f:
            json.dump(self.llm_outputs, f, indent=4)
        
        return llm_output


    #--------------------------send-request-to-pose-commander-and-await-response------------------------
    def send_req_with_planner_node(self, req):
        planner_client = PlannerClient()
        planner_client.get_logger().info("planner client node created")
        result = planner_client.send_request(req)
        planner_client.destroy_node()
        return result

    #----------------------------callback-functions------------------------------
    def user_input_callback(self, request, response):
        self.get_logger().info("using the callback")
        self.get_logger().info(f"recieved msg from user: {request.user_input}")
    
        if self.item_dict:
            self.get_logger().info("invoking chain")
            msg_to_llm_chain = self.chat_template.format_messages(function_list = self.tools_str, task = request)
            result = self.chain.invoke(msg_to_llm_chain)
            response.success = True
            response.msg = result
            
        else:
            self.get_logger().info("waiting for item dict")
            response.success = False
            response.msg = "no item dict available"
        
        return response
    
    def item_dict_callback(self, msg):
        self.get_logger().info(f"item dict recived with length {len(msg.data)}")
        self.item_dict = msg.data
        global item_dict_global
        item_dict_global = json.loads(self.item_dict)
        #self.get_logger().info(f"new global item dict {item_dict_global}")

    #we need to add gripper state callback 
    def right_robot_position_callback(self, msg):
        global robot_full_pose
        robot_full_pose['step_1']['right_ee_coor']=[msg.position.x,
                                                    msg.position.y,
                                                    msg.position.z]
        
        robot_full_pose['step_1']['right_ee_orientation'] = [msg.orientation.x,
                                                             msg.orientation.y,
                                                             msg.orientation.z,
                                                             msg.orientation.w]
        #self.get_logger().info(f"right robot pose recieved, new ee pos {robot_full_pose['step_1']['right_ee_coor']}")


    def left_robot_position_callback(self, msg):
        global robot_full_pose
        robot_full_pose['step_1']['left_ee_coor']= [msg.position.x,
                                                    msg.position.y,
                                                    msg.position.z]
        
        robot_full_pose['step_1']['left_ee_orientation'] = [msg.orientation.x,
                                                             msg.orientation.y,
                                                             msg.orientation.z,
                                                             msg.orientation.w]
        #self.get_logger().info(f"left robot pose recieved, new ee pos {robot_full_pose['step_1']['left_ee_coor']}")

    #-----------------------functions-used-by-LLM--------------------------------
   

    
def main(args=None):
    # initialize the ROS communication
    rclpy.init(args=args)
    # declare the node constructor

    gpt_controller = GptController()

    rclpy.spin(gpt_controller)


    # Explicity destroy the node
    gpt_controller.destroy_node()
    # shutdown the ROS communication
    rclpy.shutdown()


if __name__ == '__main__':
    main()
