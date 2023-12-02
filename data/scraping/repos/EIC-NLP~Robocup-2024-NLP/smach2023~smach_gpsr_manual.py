# Flow of the GPSR task (General Purpose Service Robot)

""" 
Main Goal: 3 commands requedted by the operator + 1 by non-expert 

Focus: 
    - Task Planning
    - Object/People Detectionm
    - object Feature Recognition
    - Object Manipulation
    - 5 mins

0. Feature used
    - ChatGPT

1. Robot moves from outside of Arena to the center of the arena(Instruction Point)
2. User gives the command (Wake Word) i.e.("Bring me a coke from the kitchen", "I want a coke, go to the kitchen and get me one")
3. Robot says "I am going to bring you a coke from the kitchen" (+100)
4. Robot goes to the kitchen and finds a coke (+400)
4. If fails, a custom message can be said (-50)


Examples commands from Rulebook:
Bring me a coke from the kitchen
I want a coke, go to the kitchen and get me one



"""

# run with conda env: nlp
import roslib
import rospy
import smach
import sys
import smach_ros
import nlp_client
from ratfin import *
import os
import openai
from person import Person
from utils import (WakeWord, Speak, GetIntent, GetName, GetObject, GetLocation,)

# Add the path to main repo folder to the environment 
sys.path.append('/home/walkie/Robocup-2023-NLP')
from config import gpt3_5_turbo_key

# Task specific state
class ChatGPTQuery(smach.State):
    def __init__(self):
        smach.State.__init__(self,
                            outcomes=['out1', 'out0'],
                            input_keys=['prompt'],
                            output_keys=['chatgpt_response'])
        self.messages = [{
            "role": "system", 
            "content" : """You’re a kind helpful assistant robot, 
                            respond back to me what my commands were but rephrase it like a assistant would 
                            by accepting my request. don't ask a question back just do as I says. For example, 
                            if I ask you to retrieve a coke. You should respond with something like "Certainly, grabing you a coke now" but make the sentence dynamic dont actually use the word certainly its too formal. 
                            This is a role-play"""}]
        openai.api_key = gpt3_5_turbo_key


    def execute(self, userdata):

        # Log the execution stage
        rospy.loginfo(f'(ChatGPTQuery): Executing..')

        # Copy the assistant mode prompt
        messages = self.messages.copy()

        prompt = userdata.prompt

        # Log the execution stage
        rospy.loginfo(f'(ChatGPTQuery): Prompt: {prompt}')

        # Append to messages
        messages.append({"role": "user", "content": prompt})
        
        # packaged the messages and post request
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        # extract the content of the post request
        chat_response = completion.choices[0].message.content

        # Log the execution stage
        rospy.loginfo(f'(ChatGPTQuery): ChatGPT response: {chat_response}')

        # # Speak the response from ChatGPT
        nlp_client.speak(chat_response)

        # Save the response to userdata
        userdata.chatgpt_response = chat_response

        

        return 'out1'

def main():
    speak_debug = False
    response_debug = False

    rospy.init_node('smach_task_gpsr')
    
    # Create a SMACH state machine
    sm = smach.StateMachine(outcomes=['out0'])
    
    # Declear Variables for top-state
    sm.userdata.intent = ""
    sm.userdata.stt_text = ""
    sm.userdata.chatgpt_response = ""
    

    with sm:
        # Navigation from the outside of the arena to the center of the arena

        # Engage Idle state with wake word
        smach.StateMachine.add('IDLE_TASK_TRIGGER',
                            WakeWord(),
                            transitions={'out1': 'GET_INTENT',}
                                        )
        
        # Get the intent/task of the user
        smach.StateMachine.add('GET_INTENT',
                               GetIntent(speak_debug=speak_debug,
                                         response_debug=response_debug,
                                         timeout=2),
                               transitions={'out1': 'GET_CHATGPT_QUERY',
                                            'out0': 'out0'},
                               remapping={'listen_intent': 'intent',
                                          'listen_text': 'stt_text'})

        # use ChatGPT 
        smach.StateMachine.add('GET_CHATGPT_QUERY',
                               ChatGPTQuery(),
                               transitions={'out1': 'out0',
                                            'out0': 'out0'},
                               remapping={'prompt': 'stt_text',
                                          'chatgpt_response': 'chatgpt_response'})


        

    # Execute SMACH plan
    outcome = sm.execute()

if __name__ == '__main__':
    main()
