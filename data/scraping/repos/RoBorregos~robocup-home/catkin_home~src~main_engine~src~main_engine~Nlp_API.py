#!/usr/bin/python3
import openai
import os
import rospy
from std_msgs.msg import Bool

openai.api_key = os.getenv("OPENAI_API_KEY")


class test(object):
    def __init__(self):
            self.GPT_model = "text-davinci-003"
            self.GPT_temperature = 0.7
            self.GPT_max_tokens = 256
            self.GPT_top_p = 1

            self.GPT_frequency_penalty = 0
            self.GPT_presence_penalty = 0

            self.saying_sub_ = rospy.Subscriber("saying", Bool, self.say_callback)
            

    def say_callback(self, data):
        self.saying = data.data
        rospy.loginfo("saying: " + str(self.saying))
            
    
    def callGPT(self, pr):

        response =openai.Completion.create(
            model=self.GPT_model,
            prompt=pr,
            temperature=self.GPT_temperature,
            max_tokens=self.GPT_max_tokens,
            top_p=self.GPT_top_p,
            frequency_penalty=self.GPT_frequency_penalty,
            presence_penalty=self.GPT_presence_penalty
            )
        print(response.choices[0].text)
        return response.choices[0].text
    

def main():
    rospy.init_node('test', anonymous=True)
    rospy.loginfo("test task created.")
    t = test()
    t.callGPT( "get me the name of the person on the next message: My name is Kevin.")

if __name__ == '__main__':
    main()