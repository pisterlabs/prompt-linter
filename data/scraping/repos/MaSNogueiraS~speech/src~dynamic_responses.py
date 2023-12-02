#!/usr/bin/env python3
# -- coding: utf-8 --

import rospy
import openai
from speech.srv import Chat, ChatResponse

class ChatService:
    def __init__(self) -> None:
        self.openai_api_key = "Your openai Key-here"  # Replace with your actual key
        openai.api_key = self.openai_api_key

    def chat(self, req):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", 
                messages=[
                    {"role": "system", "content": req.context},
                    {"role": "user", "content": req.user_input}
                ]
            )
            return ChatResponse(response=response.choices[0].message['content'])
        except Exception as e:
            rospy.logerr("Failed to generate response: %s", str(e))
            return ChatResponse(response="")

if __name__ == "__main__":
    rospy.init_node('chat_service_node')
    cs = ChatService()
    s = rospy.Service('chat_service', Chat, cs.chat)
    rospy.spin()




#EX of use :

# #!/usr/bin/env python3

# import rospy
# from your_package.srv import Chat

# def chat_call():
#     rospy.init_node('chat_call_node')
#     rospy.wait_for_service('chat_service')
#     try:
#         chat_service = rospy.ServiceProxy('chat_service', Chat)
#         context = "You are a helpful assistant."
#         user_input = "Hello, my name is Hera. How can I assist you today?"
#         result = chat_service(context, user_input)
#         print("Received response: " + result.response)
#     except rospy.ServiceException as e:
#         print("Service call failed: %s"%e)

# if __name__ == "__main__":
#     chat_call()
