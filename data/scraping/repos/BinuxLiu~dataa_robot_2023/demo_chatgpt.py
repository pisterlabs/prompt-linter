#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#enconding = utf8

import sys
import time
import grpc
import io
import random
from pathlib import Path

from openai import OpenAI
import pygame
import pyaudio
import wave
import keyboard

import asone
from asone import utils
from asone import ASOne

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append('./')
sys.path.append('../')

import GrabSim_pb2_grpc
import GrabSim_pb2


detector = ASOne(detector=asone.YOLOV7_PYTORCH, use_cuda=True) # Set use_cuda to False for cpu
filter_classes = ['person'] # Set to None to detect all classes

channel = grpc.insecure_channel('localhost:30001',options = [
            ('grpc.max_send_message_length', 1024*1024*1024),
            ('grpc.max_receive_message_length', 1024*1024*1024)
        ])

sim_client = GrabSim_pb2_grpc.GrabSimStub(channel)

def Init():
    sim_client.Init(GrabSim_pb2.NUL())

def AcquireAvailableMaps():
    AvailableMaps = sim_client.AcquireAvailableMaps(GrabSim_pb2.NUL())

def SetWorld(map_id = 0, scene_num = 1):
    print('------------------SetWorld----------------------')
    world = sim_client.SetWorld(GrabSim_pb2.BatchMap(count = scene_num, mapID = map_id))

def Observe(scene_id=0):
    print('------------------show_env_info----------------------')
    scene = sim_client.Observe(GrabSim_pb2.SceneID(value=scene_id))
    print(
        f"location:{[scene.location]}, rotation:{scene.rotation}\n",
        f"joints number:{len(scene.joints)}, fingers number:{len(scene.fingers)}\n", 
        f"objects number: {len(scene.objects)}, walkers number: {len(scene.walkers)}\n"
        f"timestep:{scene.timestep}, timestamp:{scene.timestamp}\n"
        f"collision:{scene.collision}, info:{scene.info}")

def Reset(scene_id = 0):
    print('------------------Reset----------------------')
    scene = sim_client.Reset(GrabSim_pb2.ResetParams(scene = scene_id))

def navigation_move(scene_id=0, map_id=0, walk_v = [247, 520, 180]):
    print('------------------navigation_move----------------------')
    scene = sim_client.Observe(GrabSim_pb2.SceneID(value=scene_id))

    walk_v = walk_v + [100, 0]
    action = GrabSim_pb2.Action(scene = scene_id, action = GrabSim_pb2.Action.ActionType.WalkTo, values = walk_v)
    scene = sim_client.Do(action)

def add_walkers(scene_id = 0, walker_loc = [50, -270], input_id = random.randint(8,12)):
    print('------------------add_walkers----------------------')
    loc = walker_loc + [0, 0, 100]
    action = GrabSim_pb2.Action(scene = scene_id, action = GrabSim_pb2.Action.ActionType.WalkTo, values = loc)
    scene = sim_client.Do(action)
    if(str(scene.info).find('unreachable') > -1):
        print("------------------This position can't used init NPCs------------------")
    else:
        walker_list = [GrabSim_pb2.WalkerList.Walker(id = input_id, pose = GrabSim_pb2.Pose(X = loc[0], Y = loc[1], Yaw = 0))]

    scene = sim_client.AddWalker(GrabSim_pb2.WalkerList(walkers = walker_list, scene = scene_id))
    return scene

def control_walkers(scene_id = 0, walker_loc = [50, 520, 0]):
    print('------------------control_walkers----------------------')
    scene = sim_client.Observe(GrabSim_pb2.SceneID(value = scene_id))
    pose = GrabSim_pb2.Pose(X=walker_loc[0], Y=walker_loc[1], Yaw=walker_loc[2])
    controls = [GrabSim_pb2.WalkerControls.WControl(id = 0, autowalk = False, speed = 160, pose = pose)]
    scene = sim_client.ControlWalkers(GrabSim_pb2.WalkerControls(controls = controls, scene = scene_id))
    return scene

def remove_walkers(scene_id = 0, id_list = [0]):
    print('------------------remove_walkers----------------------')
    remove_id_list = id_list
    scene = sim_client.RemoveWalkers(GrabSim_pb2.RemoveList(IDs = remove_id_list, scene=scene_id))
    return scene

def clean_walkers(scene_id = 0):
    print('------------------clean_walkers----------------------')
    scene = sim_client.CleanWalkers(GrabSim_pb2.SceneID(value = scene_id))
    return scene

def talk_walkers(scene_id = 0, input_content = ""):
    # print('------------------talk_walkers----------------------')
    scene = sim_client.Observe(GrabSim_pb2.SceneID(value = scene_id))
    if len(scene.walkers) > 0:
        walker_name = scene.walkers[0].name
        talk_content = walker_name + ":" + input_content
        control_robot_action(0, 0, 3, talk_content)
        return True


def get_camera(part, scene_id=0):
    print('------------------get_camera----------------------')
    action = GrabSim_pb2.CameraList(cameras=part, scene=scene_id)
    return sim_client.Capture(action)


def detect_and_show(img_data):
    print('------------------show_image----------------------')
    im = img_data.images[0]
    frame = np.frombuffer(im.data, dtype=im.dtype).reshape((im.height, im.width, im.channels))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    dets, img_info = detector.detect(frame, filter_classes=filter_classes)
    bbox_xyxy = dets[:, :4]
    class_ids = dets[:, 5]
    frame = utils.draw_boxes(frame, bbox_xyxy, class_ids=class_ids)
    # cv2.imwrite("./object_detection.jpg", frame)
    roi = None
    for bbox, class_id in zip(bbox_xyxy, class_ids):
        x1, y1, x2, y2 = map(int, bbox)
        
        roi = frame[y1:y2, x1:x2]

    print("------------------按下 q 退出检测可视化。------------------")
    while True:
        cv2.imshow('Detection Result', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    if 0 in class_ids:
        detect_result = True
    else:
        detect_result = False

    return roi, detect_result

def move_task_area(task_type = 0, scene_id = 0):
    scene = sim_client.Observe(GrabSim_pb2.SceneID(value = scene_id))

    # print('------------------move_task_area----------------------')

    if task_type == 1:  
        v_list = [[250.0, 310.0]]
    elif task_type == 2:  
        v_list = [[-70.0, 480.0]]
    elif task_type == 3:  
        v_list = [[250.0, 630.0]]
    elif task_type == 4:  
        v_list = [[-70.0, 740.0]]
    elif task_type == 5:  
        v_list = [[260.0, 1120.0]]
    elif task_type == 6:  
        v_list = [[300.0, -220.0]]
    elif task_type == 7:  
        v_list = [[0.0, -70.0]]
    else:
        v_list = [[0, 0]]
        
    for walk_v in v_list:
        walk_v = walk_v + [scene.rotation.Yaw, 60, 0]
        action = GrabSim_pb2.Action(scene=scene_id, action=GrabSim_pb2.Action.ActionType.WalkTo, values=walk_v)
        scene = sim_client.Do(action)

def control_robot_action(scene_id = 0, type = 0, action = 0, message = "你好"):
    scene = sim_client.ControlRobot(GrabSim_pb2.ControlInfo(scene = scene_id, type = type, action = action, content = message))
    if(str(scene.info).find("Action Success") > -1):
        return True
    
    else:
        return False

def record_audio(duration=10):
    output_file = 'recorded_audio.wav'
    audio = pyaudio.PyAudio()

    stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=16000, input=True,
                        frames_per_buffer=1024)

    print("按下空格键开始录制...")
    keyboard.wait('space')

    frames = []

    while True:
        data = stream.read(1024)
        frames.append(data)
        
        if keyboard.is_pressed('q'):
            print("结束音频...")
            break
        
    stream.stop_stream()
    stream.close()
    audio.terminate()
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))


def pygame_play(content):
    pygame.mixer.init()
    audio_stream = io.BytesIO(content)
    pygame.mixer.music.load(audio_stream)

    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


def ReID(query):
    image_boss = cv2.imread("./demo/boss.jpg")
    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(image_boss, None)
    keypoints2, descriptors2 = sift.detectAndCompute(query, None)

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # result = cv2.drawMatches(image_boss, keypoints1, query, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # cv2.imshow('Feature Matching', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return (len(good_matches) / len(keypoints1))

def rotate_joints(scene_id = 0, action_list = []):
    print('------------------rotate_joints----------------------')

    for values in action_list:
        action = GrabSim_pb2.Action(scene = scene_id, action = GrabSim_pb2.Action.ActionType.RotateJoints, values = values)
        scene = sim_client.Do(action)


def reset_joints(scene_id=0):
    print('------------------reset_joints----------------------')
    values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    action = GrabSim_pb2.Action(scene = scene_id, action = GrabSim_pb2.Action.ActionType.RotateJoints, values = values)
    scene = sim_client.Do(action)



def rotate_fingers(scene_id=0):
    print('------------------rotate_fingers----------------------')
    values = [-6, 0, 45, 45, 45, -6, 0, 45, 45, 45]
    action = GrabSim_pb2.Action(scene = scene_id, action = GrabSim_pb2.Action.ActionType.Finger, values = values)
    scene = sim_client.Do(action)



def reset_fingers(scene_id=0):
    print('------------------reset_fingers----------------------')
    values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    action = GrabSim_pb2.Action(scene = scene_id, action = GrabSim_pb2.Action.ActionType.Finger, values = values)
    scene = sim_client.Do(action)


def demo_coffee():
    move_task_area(1)
    control_robot_action(0, 0, 1, "制作咖啡中")
    result = control_robot_action(0, 1, 1)
    control_robot_action(0, 0, 2)
    if(result):
        control_robot_action(0, 1, 2)
        control_robot_action(0, 0, 2)
        control_robot_action(0, 1, 3)
        control_robot_action(0, 0, 2)
        control_robot_action(0, 1, 4)
        control_robot_action(0, 0, 2)
        control_robot_action(0, 0, 1, "您的咖啡制作完毕。")
        navigation_move()

def demo_food():
    move_task_area(3)
    control_robot_action(0, 0, 1, "开始夹点心")
    result = control_robot_action(0, 3, 1)
    control_robot_action(0, 0, 2)
    if(result):
        control_robot_action(0, 3, 2)
        control_robot_action(0, 0, 2)
        control_robot_action(0, 3, 3)
        control_robot_action(0, 0, 2)
        control_robot_action(0, 3, 4)
        control_robot_action(0, 0, 2)
        control_robot_action(0, 3, 5)
        control_robot_action(0, 0, 2)
        control_robot_action(0, 3, 6)
        control_robot_action(0, 0, 2)
        control_robot_action(0, 3, 7)
        control_robot_action(0, 0, 2)
        control_robot_action(0, 0, 1, "夹点心结束")
        navigation_move()

def demo_water():
    move_task_area(2)
    control_robot_action(0, 0, 1, "开始倒水")
    result = control_robot_action(0, 2, 1)
    control_robot_action(0, 0, 2)
    if(result):
        control_robot_action(0, 2, 2)
        control_robot_action(0, 0, 2)
        control_robot_action(0, 2, 3)
        control_robot_action(0, 0, 2)
        control_robot_action(0, 2, 4)
        control_robot_action(0, 0, 2)
        control_robot_action(0, 2, 5)
        control_robot_action(0, 0, 2)
        control_robot_action(0, 0, 1, "倒水结束")
        navigation_move()

def demo_clear():
    move_task_area(4)
    control_robot_action(0, 0, 1, "开始拖地")
    result = control_robot_action(0, 4, 1)
    control_robot_action(0, 0, 2)
    if(result):
        control_robot_action(0, 4, 2)
        control_robot_action(0, 0, 2)
        control_robot_action(0, 4, 3)
        control_robot_action(0, 0, 2)
        control_robot_action(0, 4, 4)
        control_robot_action(0, 0, 2)
        control_robot_action(0, 0, 1, "拖地结束")
        control_walkers(0, [50, 520, 0])
        time.sleep(3.0)
        navigation_move()


def demo_close_curtain():
        control_robot_action(0, 0, 1, "关闭窗帘")
        result = control_robot_action(0, 8, 1)
        control_robot_action(0, 0, 2)
        if(result):
            control_robot_action(0, 0, 1, "关闭窗帘成功")


def demo_open_ac():
    navigation_move(0, map_id, walk_v = [249.0, -155.0, 0])
    time.sleep(1.0)
    rotate_joints(0, [[0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 36.0, -39.37, 37.2, -92.4, 4.13, -0.62, 0.4],
                        [0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 36.0, -39.62, 34.75, -94.80, 3.22, -0.26, 0.85],
                        [0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 32.63, -32.80, 15.15, -110.70, 6.86, 2.36, 0.40],
                        [0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 28.18, -27.92, 6.75, -115.02, 9.46, 4.28, 1.35],
                        [0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 4.09, -13.15, -11.97, -107.35, 13.08, 8.58, 3.33]])
    time.sleep(1.0)
    rotate_fingers(0)
    time.sleep(1.0)
    rotate_joints(0, [[0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, -29.09, -20.15, -11.97, -70.35, 13.08, 8.58, 3.33],
                        [0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0, -30.09, -19.15, -11.97, -70.35, 13.08, 8.58, 3.33]])
    time.sleep(1.0)
    reset_joints(0)
    time.sleep(1.0)
    reset_fingers(0)
    navigation_move(0, map_id, walk_v = [247, 520, 180])

if __name__ == '__main__':

    map_id = 11
    scene_num = 1
    
    OPENAI_SPEECH = True
    from openai_key import DEMO_KEY
    openai_client = OpenAI(api_key=DEMO_KEY,)

    MESSAGE_1 = "你是咖啡店的服务生，我是顾客。"
    
    MESSAGE_2 = "你是咖啡点的服务生，我是店长。"
    
    MESSAGE_3 = "你有以下商品：{[名称:'做咖啡', 代码: '1'], [名称:'白开水', 代码: '2'],[名称:'点心', 代码: '3']}.\
                 你有以下工作需要完成：{[名称: '拖地', 代码: '4'],[名称:'擦桌子', 代码: '5'],[名称:'关窗帘', 代码: '6'],[名称:'关灯', 代码: '7'],[名称:'调节空调', 代码: '8']}\
                 如果我没有具体商品或任务被指定，你回复格式如下: '内容'。\
                 如果我指定了具体商品或任务，你回复格式如下: '内容' + '#代码'。\
                 如果我的语句中包含感谢你的服务,你回复'再见'或'期待为您的下次服务'。\
                 示例，顾客：你们店里有什么呢？服务员：我们店里有咖啡，白开水，蛋糕。\
                      顾客：给我来一个蛋糕。服务员：好的，我这就给你来一个小蛋糕。#3\
                      顾客：再给我来一杯咖啡吧。服务员：请稍等，一杯美味的咖啡正在制作中。#1\
                      店长：你今天拖地了吗？服务员：您觉得不干净吗？我这就再拖一次。#4\
                      顾客：我有点冷，可以开空调吗？服务员：好的，我这就去打开空调。#8"


    Init()
    SetWorld(map_id, scene_num)
    time.sleep(5.0)
    
    add_walkers(0)
    # add_walkers(0, [50, -270], 13)
    time.sleep(1.0)
    control_walkers(0)
    time.sleep(10.0)

    img_data = get_camera([GrabSim_pb2.CameraName.Head_Color], 0)
    roi, detect_result = detect_and_show(img_data)

    times = 0
    try:
        while detect_result:

            if ReID(roi) < 0.2:
                massage = MESSAGE_1 + MESSAGE_3
                messages = [{"role": "system", "content": massage},]
                hello_world = "您好，欢迎光临！有什么可以为您服务的？"
                messages.append({"role": "assistant", "content": hello_world})
                control_robot_action(0, 0, 1, hello_world)
                boss_mode = False
                
            else:
                massage = MESSAGE_2 + MESSAGE_3
                messages = [{"role": "system", "content": massage},]
                hello_world = "您好，店长！有什么指示？"
                messages.append({"role": "assistant", "content": hello_world})
                control_robot_action(0, 0, 1, hello_world)
                boss_mode = True
            print(hello_world)
            if OPENAI_SPEECH:
                tts_response = openai_client.audio.speech.create(
                    model="tts-1", voice="nova", input=hello_world
                )
                pygame_play(tts_response.content)

            while True:
                if OPENAI_SPEECH:
                    record_audio()
                    transcriptions = openai_client.audio.transcriptions.create(model="whisper-1", file=Path(__file__).parent / "recorded_audio.wav",)
                    user_input = transcriptions.text
                else:
                    user_input = input()
                talk_walkers(0, user_input)
                messages.append({"role": "user", "content": user_input})

                completion = openai_client.chat.completions.create(model="gpt-3.5-turbo", 
                                                                   messages=messages,
                                                                   max_tokens=60)

                gpt_response = completion.choices[0].message.content
                if "#" in gpt_response:
                    gpt_codes = gpt_response.split("#")[1]
                    gpt_response = gpt_response.split("#")[0]

                    if gpt_codes == "6" or gpt_codes == "7":
                        if boss_mode == False:
                            gpt_response = "不好意思，您没有店长权限"
                    control_robot_action(0, 0, 1, gpt_response)
                    print(gpt_response)
                    if OPENAI_SPEECH:
                        tts_response = openai_client.audio.speech.create(
                                    model="tts-1", voice="nova", input=gpt_response)
                    
                        pygame_play(tts_response.content)

                    if gpt_codes == "1":
                        demo_coffee()
                    elif gpt_codes == "2":
                        control_walkers(0, [0, 540, 180])
                        time.sleep(5.0)
                        demo_water()
                        control_walkers(0, [0, 540, 0])
                        time.sleep(5.0)
                    elif gpt_codes == "3":
                        demo_food()
                    elif gpt_codes == "4":
                        if boss_mode == True:
                            control_walkers(0, [247, 470, 180])
                            time.sleep(8.0)
                            demo_clear()
                    elif gpt_codes == "6":
                        if boss_mode == "True":
                            demo_close_curtain()
                    elif gpt_codes == "8":
                        demo_open_ac()
                            


                else:
                    control_robot_action(0, 0, 1, gpt_response)
                    print(gpt_response)
                    if OPENAI_SPEECH:
                        tts_response = openai_client.audio.speech.create(
                                    model="tts-1", voice="nova", input=gpt_response)
                        pygame_play(tts_response.content)

 

                if "再见" in gpt_response or "期待为您的下次服务" in gpt_response:
                    break

                messages.append({"role": "assistant", "content": gpt_response})

            control_walkers(0, [50, -270, 0])
            time.sleep(8.0)
            clean_walkers()
            Reset(0)
            if times == 0:
                add_walkers(0, [50, -270])
                time.sleep(1.0)
                control_walkers(0)
                time.sleep(10.0)
            elif times == 1:
                add_walkers(0, [50, -270], 13)
                time.sleep(1.0)
                control_walkers(0)
                time.sleep(10.0)
            times = times + 1

            img_data = get_camera([GrabSim_pb2.CameraName.Head_Color], 0)
            roi, detect_result = detect_and_show(img_data)

    except KeyboardInterrupt:
        Reset(0)
        clean_walkers()
        print("\n展示结束。")



        

