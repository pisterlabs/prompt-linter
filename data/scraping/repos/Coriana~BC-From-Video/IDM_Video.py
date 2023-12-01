# NOTE: this is _not_ the original code of IDM!
# As such, while it is close and seems to function well,
# its performance might be bit off from what is reported
# in the paper.

from argparse import ArgumentParser
import pickle
import cv2
import numpy as np
import json
import torch as th
import ffmpeg 

from openai_vpt.agent import ENV_KWARGS, resize_image, AGENT_RESOLUTION
from inverse_dynamics_model import IDMAgent


KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape" :"ESC",
    "key.keyboard.s" :"back",
    "key.keyboard.q" :"drop",
    "key.keyboard.w" :"forward",
    "key.keyboard.1" :"hotbar.1",
    "key.keyboard.2" :"hotbar.2",
    "key.keyboard.3" :"hotbar.3",
    "key.keyboard.4" :"hotbar.4",
    "key.keyboard.5" :"hotbar.5",
    "key.keyboard.6" :"hotbar.6",
    "key.keyboard.7" :"hotbar.7",
    "key.keyboard.8" :"hotbar.8",
    "key.keyboard.9" :"hotbar.9",
    "key.keyboard.e" :"inventory",
    "key.keyboard.space" :"jump",
    "key.keyboard.a" :"left",
    "key.keyboard.d" :"right",
    "key.keyboard.left.shift" :"sneak",
    "key.keyboard.left.control" :"sprint",
    "key.keyboard.f" :"swapHands",
}

# Template action
NOOP_ACTION = {
    "ESC": 0,
    "back": 0,
    "drop": 0,
    "forward": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "inventory": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    "camera": np.array([0, 0]),
    "attack": 0,
    "use": 0,
    "pickItem": 0,
}

MESSAGE = """
This script will take a video, predict actions for its frames and
and show them with a cv2 window.

Press any button the window to proceed to the next frame.
"""

# Matches a number in the MineRL Java code regarding sensitivity
# This is for mapping from recorded sensitivity to the one used in the model
CAMERA_SCALER = 360.0 / 2400.0



def cam_to_pix(c):
  return int(c/10.0 * 66.5) # Linear relationship(!) from earlier calibration tests
  
def main(video_path):
    model = '4x_idm.model'
    weights = '4x_idm.weights'
    print(MESSAGE)
    agent_parameters = pickle.load(open(model, "rb"))
    net_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = IDMAgent(idm_net_kwargs=net_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w,h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    mid_x, mid_y = int(w//2), int(h//2)
    frame_size = (640, 360)
    outframe_size = (128,128)

    # Obtain frame size information using get() method
    # Initialize video writer object
    prev_kb, prev_buttons, new_kb, new_buttons, kb , buttons = [], [],[], [],[], [],
    started = 0
    framenum = 0
    vidnum = 1
    FutureFrames = []
    out_path = video_path[:-4] # remove .mp4
    output = cv2.VideoWriter(out_path + str(vidnum) + '-debug.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, frame_size)
    clean_output = cv2.VideoWriter(out_path + str(vidnum) + '_.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, outframe_size)
    f = open(out_path + str(vidnum) + '_.jsonl', 'a') # make .jsonl file and open for writing
    while(cap.isOpened()):
    
        th.cuda.empty_cache()
        print("=== Loading up frames ===")
        # Copy 2nd 1/2 of previous buffer to current buffer & clear future buffer.
        CurrentFrames = FutureFrames  
        FutureFrames = []

        for _ in range(32):
            ret, frame = cap.read()
            if not ret:
                break

            CurrentFrames.append(frame[..., ::-1]) # Add frame to current buffer, to fill to 128 frames for proper prediction.
            FutureFrames.append(frame[..., ::-1]) # Add frame to future buffer, so 2nd half can be used next batch.
            
        Num_frames = len(CurrentFrames)
        
        if Num_frames != 64:
            if started == 0:
                started = 1
                print("=== Not full batch ===")
                continue
            else:
                break
        else:
            frames = np.stack(CurrentFrames)
            print("=== Predicting actions ===")
            predicted_actions = agent.predict_actions(frames)

            for i in range(Num_frames):
                # first 32 frames are untrusted, but don't worry, these actions were predicted last batch, or are from the first 32 frames of the video. 
                if i < 16: 
                    continue
                    
                if i > 48: # untrusted as not enough future data is known to predict the actions, don't worry, next batch will have these as the first 32 trusted predictions.
                    continue
                
                framenum= framenum +1
                
                frame = frames[i]
                outframe = resize_image(frame[..., ::-1], outframe_size)
                clean_output.write(outframe)
                for y, (action_name, action_array) in enumerate(predicted_actions.items()):
                    current_prediction = action_array[0, i]
                    if action_name == 'camera':
                        cam_in = action_array[0, i]
                        cam_1 = cam_in[:-1]     #drop 1/2
                        cam_2 = cam_in[-1:]     #drop the other 1/2
                        #cam=f"cam[{a['cam_in'][0][0]:+7.2f}, {a['cam_in'][0][1]:+7.2f}]"
                        #print(cam_2)
                    else:
                        if action_array[0, i] == 1:  
                            #kb.append(action_name)                            
                            cv2.putText(
                                frame,
                                f"{action_name}",
                                (10, 25 + y * 18),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 170, 0),
                                2
                            )
                            if action_name == 'back':
                                kb.append("key.keyboard.s") 
                                if "key.keyboard.s" not in prev_kb:
                                    new_kb.append("key.keyboard.s")
                                
                            if action_name == 'forward':
                                kb.append("key.keyboard.w") 
                                if "key.keyboard.w" not in prev_kb:
                                    new_kb.append("key.keyboard.w")
                                    
                            if action_name == 'drop':
                                kb.append("key.keyboard.q") 
                                if "key.keyboard.q" not in prev_kb:
                                    new_kb.append("key.keyboard.q")
                                    
                            if action_name == 'inventory':
                                kb.append("key.keyboard.e") 
                                if "key.keyboard.e" not in prev_kb:
                                    new_kb.append("key.keyboard.e")
                                    
                            if action_name == 'jump':
                                kb.append("key.keyboard.space") 
                                if "key.keyboard.space" not in prev_kb:
                                    new_kb.append("key.keyboard.space")
                                    
                            if action_name == 'left':
                                kb.append("key.keyboard.a") 
                                if "key.keyboard.a" not in prev_kb:
                                    new_kb.append("key.keyboard.a")
                                    
                            if action_name == 'right':
                                kb.append("key.keyboard.d") 
                                if "key.keyboard.d" not in prev_kb:
                                    new_kb.append("key.keyboard.d")
                                    
                            if action_name == 'sneak':
                                kb.append("key.keyboard.left.shift") 
                                if "key.keyboard.left.shift" not in prev_kb:
                                    new_kb.append("key.keyboard.left.shift")
                                    
                            if action_name == 'sprint':
                                kb.append("key.keyboard.left.control") 
                                if "key.keyboard.left.control" not in prev_kb:
                                    new_kb.append("key.keyboard.left.control")
                                    
                            if action_name == 'hotbar.1':
                                kb.append("key.keyboard.1") 
                                if "key.keyboard.1" not in prev_kb:
                                    new_kb.append("key.keyboard.1")
                            
                            if action_name == 'hotbar.2':
                                kb.append("key.keyboard.2") 
                                if "key.keyboard.2" not in prev_kb:
                                    new_kb.append("key.keyboard.2")
                            
                            if action_name == 'hotbar.3':
                                kb.append("key.keyboard.3") 
                                if "key.keyboard.3" not in prev_kb:
                                    new_kb.append("key.keyboard.3")
                                    
                            if action_name == 'hotbar.4':
                                kb.append("key.keyboard.4") 
                                if "key.keyboard.4" not in prev_kb:
                                    new_kb.append("key.keyboard.4")
                                    
                            if action_name == 'hotbar.5':
                                kb.append("key.keyboard.5") 
                                if "key.keyboard.5" not in prev_kb:
                                    new_kb.append("key.keyboard.5")
                                    
                            if action_name == 'hotbar.6':
                                kb.append("key.keyboard.6") 
                                if "key.keyboard.6" not in prev_kb:
                                    new_kb.append("key.keyboard.6")
                                    
                            if action_name == 'hotbar.7':
                                kb.append("key.keyboard.7") 
                                if "key.keyboard.7" not in prev_kb:
                                    new_kb.append("key.keyboard.7")
                                    
                            if action_name == 'hotbar.8':
                                kb.append("key.keyboard.8") 
                                if "key.keyboard.8" not in prev_kb:
                                    new_kb.append("key.keyboard.8")
                                    
                            if action_name == 'hotbar.9':
                                kb.append("key.keyboard.9") 
                                if "key.keyboard.9" not in prev_kb:
                                    new_kb.append("key.keyboard.9")
                                    
                            
                            if action_name == 'attack':
                                buttons.append(0) 
                                if 0 not in prev_buttons:
                                    new_buttons.append(0)
                                    
                            if action_name == 'Use':
                                buttons.append(1) 
                                if 1 not in prev_buttons:
                                    new_buttons.append(1)
                                    
                                
                                
                pix_pitch, pix_yaw = cam_to_pix(cam_1), cam_to_pix(cam_2)
                frame = cv2.arrowedLine(frame, (mid_x,mid_y), (mid_x+pix_yaw, mid_y+pix_pitch), (255, 170, 0), 5)
                                    
                prediction = '{"mouse":{"dx":' + str(int(cam_2/CAMERA_SCALER)) + ',"dy":'+ str(int(cam_1/CAMERA_SCALER)) +',"buttons":'+ str(buttons) +', "newButtons": '+ str(new_buttons) +'},"keyboard":{"keys":'+ str(kb) +', "newKeys": '+ str(new_kb) +'},"isGuiOpen":false}'
                prev_kb, prev_buttons = kb , buttons
                kb, new_kb , buttons, new_buttons = [], [], [], []

                # print(prediction)
                prediction = str.replace(prediction, chr(39), chr(34))
                f.write(prediction + '\n') # write to file
                outframe = resize_image(frame[..., ::-1], frame_size)
                output.write(outframe)
                if framenum > 12000:
                    vidnum = vidnum +1
                    output.release()
                    f.close()
                    clean_output.release()
                    output = cv2.VideoWriter(out_path + str(vidnum) + '-debug.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, frame_size)
                    clean_output = cv2.VideoWriter(out_path + str(vidnum) + '_.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, outframe_size)
                    f = open(out_path + str(vidnum) + '_.jsonl', 'a')
                    framenum = 0
                #cv2.imwrite('frame'+ str(framenum) + '.png', frame[..., ::-1])
                #cv2.imshow("MineRL IDM model predictions", frame[..., ::-1])
                #cv2.waitKey(0)
                
    print(f"Fames left unprocessed at end of the video: {Num_frames}")
    cap.release()
    output.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    parser = ArgumentParser("Run IDM on MineRL recordings.")

    parser.add_argument("--video-path", type=str, required=True, help="Path to a .mp4 file (Minecraft recording).")

    args = parser.parse_args()

    main(args.video_path)
