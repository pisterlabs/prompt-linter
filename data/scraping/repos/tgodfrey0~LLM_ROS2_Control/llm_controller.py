import cv2
import apriltag
import openai

from SwarmNet import swarmnet
from openai import OpenAI

at_options = apriltag.DetectorOptions(families="tag36h11")
tag_width = 10
global_conv = []

def entry():
  cv2.namedWindow("Stream")
  vc = cv2.VideoCapture(0)
  
  at_detector = apriltag.Detector(at_options)

  if vc.isOpened():
      rval, frame = vc.read()
  else:
      rval = False

  while rval:
      grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      
      tags = at_detector.detect(grayscale)
      n_tags = len(tags)
      
      print(f"{n_tags} tags found")
      
      for tag in tags:
        (ptA, ptB, ptC, ptD) = tag.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))

        cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
        cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
        cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
        cv2.line(frame, ptD, ptA, (0, 255, 0), 2)
        
        (cX, cY) = (int(tag.center[0]), int(tag.center[1]))
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
        
        tagFamily = tag.tag_family.decode("utf-8")
        cv2.putText(frame, tagFamily, (ptA[0], ptA[1] - 15),
          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
          
      cv2.imshow("Stream", frame)
      rval, frame = vc.read()
      key = cv2.waitKey(20)
      if key == 27: # exit on ESC
          break
  
  cv2.destroyWindow("Stream")
  vc.release()
  
def send_req(client: OpenAI) -> str:
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=global_conv,
    max_tokens=300
  )

  # print(completion.choices[0].message)
  
  return completion.choices[0].message.content
  
def get_api_key() -> str:
  with open("openai_key", "r") as f:
    return f.readline()
  
def input_recv(msg: str) -> None: 
  print(msg)
  print("Add to dictionary or something")

if __name__=="__main__":
  sn_ctrl = swarmnet.SwarmNet({"LLM": input_recv})
  sn_ctrl.start()
  client = OpenAI(api_key=get_api_key())
  # entry()
  # global_conv = [
  #   {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."}, 
  #   {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  # ]
  global_conv = [
    {"role": "system", "content": "You are a wheeled robot, and can only move forwards, backwards, and rotate clockwise or anticlockwise. You will negotiate with other robots to navigate a path without colliding. You should negotiate and debate the plan until all agents agree. Once this has been decided you should call the '@SUPERVISOR' tag at the end of your plan."}, 
    {"role": "user", "content": "Create a plan to move on a chess board from B7 to F7 without colliding with the agent at D7"}
  ]
  res = send_req(client)
  print(res)
  sn_ctrl.send(f"LLM {res}")
  sn_ctrl.kill()