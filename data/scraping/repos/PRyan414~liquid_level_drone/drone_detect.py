import cv2
import torch
# import openai
from djitellopy import Tello
#import azure.cognitiveservices.speech as speechsdk
from pynput import keyboard
import threading
import time
import datetime
from drone_utils import DroneUtils
import configparser
import os
import speech_recognition as sr
import numpy as np
import json
import argparse

import sys
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

#########################
# SETUP
#########################

# Constants
config = configparser.ConfigParser()
config.read('config.ini')
# AZURE_SUBSCRIPTION_KEY = config.get('API key', 'AZURE_SUBSCRIPTION_KEY')
# AZURE_SERVICE_REGION = config.get('API key', 'AZURE_SERVICE_REGION')

TELLO_IP = config.get('tello', 'ip')

# Paths (change these paths as per your system)
exp = "exp_500"
root_path =  "/Users/richtsai1103/liquid_level_drone"
model_name = "best_small640"
weights_path = os.path.join(root_path, f"yolov5/runs/train/{exp}/weights/{model_name}.pt")
model_path = os.path.join(root_path, "yolov5/")

# ACTIONS TO COMMANDS MAPPING
ACTIONS_TO_COMMANDS = {
    ("start", "fly", "take off", "lift off", "launch", "begin flight", "skyward"): "takeoff",
    ("land", "settle", "touch down", "finish", "end flight", "ground"): "land",
    ("front flip", "forward flip"): "flip",
    ("forward", "move ahead", "go straight", "advance", "head forward", "proceed front", "go on", "move on"): "move_forward",
    ("backward", "move back", "retreat", "go backward", "back up", "reverse", "recede"): "move_back",
    ("left", "move left", "go leftward", "turn leftward", "shift left", "sidestep left"): "move_left",
    ("right", "move right", "go rightward", "turn rightward", "shift right", "sidestep right"): "move_right",
    ("move up", "up", "ascend", "rise", "climb", "skyrocket", "soar upwards", "elevate"): "move_up",
    ("move down", "down", "descend", "lower", "sink", "drop", "fall", "decline"): "move_down",
    ("spin right", "rotate clockwise", "turn right", "twirl right", "circle right", "whirl right", "swirl right"): "rotate_clockwise",
    ("spin left", "rotate counter-clockwise", "turn left", "twirl left", "circle left", "whirl left", "swirl left"): "rotate_counter_clockwise",
    ("back flip", "flip back"): "flip_backward",
    ("flip", "forward flip", "flip forward"): "flip_forward",
    ("right flip", "flip to the right", "sideways flip right"): "flip_right",
    ("video on", "start video", "begin stream", "camera on"): "streamon",
    ("video off", "stop video", "end stream", "camera off"): "streamoff",
    ("go xyz", "specific move", "exact move", "precise direction", "navigate xyz"): "go_xyz_speed",
    ("give me stats", "status"): "status",
    ("stop"): "disconnect"
}

drone_control_kb = {
    'move': "",
    'takeoff': False,
    'land': False,
    'navigation': False
}
        

#########################
# FUNCTIONS
#########################

def interpret_command_to_drone_action(command):
    for action_phrases, action in ACTIONS_TO_COMMANDS.items():
        if command in action_phrases:
            return action
    return None

def mock_execute_drone_command(command):
    print(f"Mock executed command: {command}")

def listen_to_commands(drone_ops, mock):
    # verbal commabd to control drone
    try:
        speech_recognizer = sr.Recognizer()
        print("Listening for commands. Speak into your microphone...")
        

        with sr.Microphone() as source:
            while True:
                print("Awaiting command...")
                audio = speech_recognizer.listen(source, phrase_time_limit = 3)
                try:
                    command_heard = speech_recognizer.recognize_google(audio).lower()
                    print(f"Heard: {command_heard}")
                except sr.UnknownValueError:
                    # keep sending signal to prevent auto shutdown
                    drone_ops.tello.send_control_command('command')
                    continue
            
                if command_heard:
                    drone_command = interpret_command_to_drone_action(command_heard)
                else:
                    print("Nothing is heard.")
            
                if drone_command:
                    print(f"\nExecuting command: {drone_command}")
                    
                    ## mocking ##
                    if mock:
                        mock_execute_drone_command(drone_command)
                    else:
                        try:
                            command = drone_ops.execute_drone_command(drone_command)
                            print(f"Executed command: {command}")
                        except Exception as e:
                            print(f"Error executing the command: {e}")
                else:
                    # keep sending signal to prevent auto shutdown
                    drone_ops.tello.send_control_command('command')
                    print(f"Not a valid action term: {command_heard}")

                # Check for a command to end the program gracefully
                if "stop" in command_heard:
                    print("Terminating program...")
                    break
                
                # sleep for buffer
                time.sleep(2)

    except Exception as e:
        print(f"Error in recognizing speech: {e}")


result_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

class CameraViewer(QMainWindow):
    def __init__(self, save_video=False, file_format='mp4'):
        super().__init__()

        self.setWindowTitle("Camera Viewer")

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        
        self.save_video = save_video
        self.file_format = file_format
        self.make_center = False
        
        # Initialize video recording attributes
        self.fps = 30.0  # Frames per second
        self.video_width = 720  # Width of the output video frame
        self.video_height = 480  # Height of the output video frame
        self.output_directory = os.path.join('video_result', exp)  # Modify 'exp' to your desired experiment name
        os.makedirs(self.output_directory, exist_ok=True)
        self.timestamp = result_timestamp
        self.output_file = f'video_{self.timestamp}' + f'.{self.file_format}'
        self.output_path = os.path.join(self.output_directory, self.output_file)
        
        if self.save_video:
            if self.file_format == 'mp4':
                self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif self.file_format == 'avi':
                self.fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format

            self.out = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.video_width, self.video_height))
        
        self.label = QLabel(self)
        self.layout.addWidget(self.label)

        self.frame_read = tello.get_frame_read()  # Open the default camera (usually index 0)
        # buffer
        time.sleep(2)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)  # Update every 1 milliseconds (adjust as needed)
        
        self.last_center_time = time.time()
        self.last_report_time = time.time()

        
    def update_frame(self):
        start_time = time.time()
        frame_original = self.frame_read.frame

        # Crop the image to make it more focued on smaller portion of the frame
        height, width, channel = frame_original.shape
        crop_x1 = width // 4  # Adjust the cropping region as needed
        crop_x2 = 3 * width // 4
        crop_y1 = height // 4
        crop_y2 = 3 * height // 4
        cropped_image = frame_original[crop_y1+20:crop_y2-20, crop_x1:crop_x2]
        # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

        # todo: change brightness
        # Increase or decrease brightness by adjusting pixel values
        brightness_factor = 1.45  # You can adjust this factor to control brightness
        cropped_image = np.clip(cropped_image * brightness_factor, 0, 255).astype(np.uint8)
        
        
        # Resize for faster processing 
        # todo: (keep the same 640 since we trained on 640 or train model in 320)
        img_size = 640
        frame_resized = cv2.resize(cropped_image, (img_size, img_size))
        
        # Set confidence level
        model.conf = 0.2
        results = model(frame_resized)
        self.rendered_frame_small = results.render()[0]
        elapsed_time = time.time() - start_time

        
        ### Reporting Section
        report_interval = 8
        if time.time() - self.last_report_time > report_interval:
            if not len(results.pred[0]):
                print('Nothing Detected')
            for detection in results.pred[0]:
                x1, y1, x2, y2, conf, class_id = map(float, detection)
                label = results.names[int(class_id)]
                print(f"Detected: {label}, Confidence: {conf:.2f}")
                
            self.last_report_time = time.time()
            print("-------------------")

        if drone_control_kb['navigation']:
            for detection in results.pred[0]:
                x1, y1, x2, y2, conf, class_id = map(float, detection)
                label = results.names[int(class_id)]
                drone_ops.p_current.append((label, conf, elapsed_time))
            
            
        ### Centering Section
        # Calculate the center of the bounding box if any object is detected
        self.make_center = drone_ops.make_center
        center_interval = 4

        try:
            if drone_control_kb['navigation'] \
                and self.make_center \
                    and (time.time() - self.last_center_time > center_interval \
                    or drone_ops.center_times == 0):
                print('\nStart Centering')
                print(f'Centering times: {drone_ops.center_times}')

                if drone_ops.center_times == 0 and time.time() - drone_ops.moving_start > 2:
                    drone_ops.center(results.pred[0][:4][0], img_size)
                    self.last_center_time = time.time()
                    drone_ops.center_times += 1
                
        except Exception:
            pass

        # Resize the rendered frame to a larger resolution for display
        rendered_frame_large = cv2.resize(self.rendered_frame_small, (self.video_width, self.video_height)) 
        rendered_frame_large = cv2.cvtColor(rendered_frame_large, cv2.COLOR_RGB2BGR)
        # Save the frame to the video file (if video recording is enabled)
        if self.save_video:
            self.out.write(rendered_frame_large)
            
        # render to QImage (since cv2 will block keyboard control)
        rendered_height, rendered_width, _ = rendered_frame_large.shape 
        bytes_per_line = 3 * rendered_width
        q_image = QImage(rendered_frame_large, rendered_width, rendered_height, bytes_per_line, QImage.Format_BGR888)

        # Display the QImage in the QLabel
        self.label.setPixmap(QPixmap.fromImage(q_image))

        
    def closeEvent(self, event):
        # Release the video writer when the application is closed
        if self.save_video:
            self.out.release()
        super().closeEvent(event)

def control_drone(mock):
    # control with kb
    while True:
        if mock:
            mock_execute_drone_command(drone_control_kb['move'])
            time.sleep(1)
        else:
            if drone_control_kb['takeoff']:
                tello.takeoff()
                drone_control_kb['takeoff'] = False
            elif drone_control_kb['land']:
                tello.land()
                drone_control_kb['land'] = False
            elif drone_control_kb['navigation']:
                drone_ops.execute_drone_command('navigation')
                drone_control_kb['navigation'] = False
            else:
                time.sleep(1)
                drone_ops.execute_drone_command(drone_control_kb['move'])
                drone_control_kb['move'] = ""
                      
# Class for keyboard control listener
class KeyboardListener(QThread):
    key_pressed = pyqtSignal(object)

    def run(self):
        def on_key_press(key):
            # Your key press logic here
            self.key_pressed.emit(key)

        with keyboard.Listener(on_press=on_key_press) as listener:
            listener.join()

# Function to handle key presses and update drone control commands
def handle_key_press(key, drone_control_kb):
    try:
        if key.char == 'w':
            drone_control_kb['move'] = 'move_forward'
        elif key.char == 's':
            drone_control_kb['move'] = 'move_back'
        elif key.char == 'a':
            drone_control_kb['move'] = 'move_left'
        elif key.char == 'd':
            drone_control_kb['move'] = 'move_right'
        elif key.char == 'e':
            drone_control_kb['move'] = 'move_up'
        elif key.char == 'q':
            drone_control_kb['move'] = 'move_down'
        elif key.char == 'k':
            drone_control_kb['move'] = 'rotate_clockwise'
        elif key.char == 'j':
            drone_control_kb['move'] = 'rotate_counter_clockwise'
        elif key.char == 'z':
            drone_control_kb['navigation'] = True
        ### This command is used to save out the experiment result after navigation
        elif key.char == 't':
            if drone_control_kb['navigation'] == False:
                t = result_timestamp
                # save out result to json
                if not os.path.exists(f"exp_result/{exp}/{model_name}"):
                    os.mkdir(f"exp_result/{exp}/{model_name}")
                with open(f"exp_result/{exp}/{model_name}/res_{t}.json", "w") as json_file:
                    json.dump(drone_ops.all_res, json_file)
            
    except AttributeError:
        if key == keyboard.Key.space:
            drone_control_kb['takeoff'] = True
        elif key == keyboard.Key.esc:
            drone_control_kb['land'] = True

    print(drone_control_kb)  # Print the updated drone control commands
    
    
    
#########################
# MAIN LOOP - Execute the script!
#########################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Control the drone with keyboard and mock options")
    parser.add_argument("--control_with_kb", action="store_true", help="Enable keyboard control")
    parser.add_argument("--mock", action="store_true", help="Use mock execution of drone commands")
    parser.add_argument("--save_video", action="store_true", help="Save video after running the experiment")

    # Parse command-line arguments
    args = parser.parse_args()
    
    # Set the control options based on parsed arguments
    control_with_kb = args.control_with_kb
    mock = args.mock
    save_video = args.save_video
    
    # Assuming you initialize drone_state as 'landed' or 'flying' elsewhere in your script
    in_flight = False
    
    # Check CUDA availability
    USE_CUDA = torch.cuda.is_available()
    DEVICE = 'cuda:0' if USE_CUDA else 'cpu'
    
    # Setup YOLOv5 with custom model weights
    model = torch.hub.load('yolov5/', 'custom', path=weights_path, source='local').to(DEVICE)
    if USE_CUDA:
        model = model.half()  # Half precision improves FPS
        print("YOLOv5 setup complete.")
        
    # --- TELLO DRONE SETUP ---
    print("Start Drone")
    tello = Tello(TELLO_IP)
    tello.connect(False)
    
    drone_ops = DroneUtils(tello, in_flight)
    # start video streaming 
    tello.streamon()
    
    # get better real-time performance
    tello.set_video_fps(tello.FPS_30)
    tello.set_video_resolution(tello.RESOLUTION_720P)
    tello.set_video_bitrate(tello.BITRATE_4MBPS)
    
    # multi-threading
    app = QApplication(sys.argv)
    viewer = CameraViewer(save_video=save_video, file_format='avi')
    viewer.show()
    
    if control_with_kb:
        drone_thread = threading.Thread(target=control_drone, args=(mock,))
        drone_thread.start()
        
        # Create a keyboard listener
        keyboard_listener = KeyboardListener()
        keyboard_listener.key_pressed.connect(lambda key: handle_key_press(key, drone_control_kb))

        # Start the keyboard listener thread
        keyboard_listener.start()
        print("Press 'w' to move forward, 's' to move backward.")
        print("Press 'a' to move left, 'd' to move right.")
        print("Press 'Space' to take off, 'Esc' to land.")
        
    else:
        listen_thread = threading.Thread(target=listen_to_commands, args=(drone_ops, mock))
        listen_thread.start()
    
    # Start the PyQt application
    sys.exit(app.exec_())
    
    # todo: fix state udp
    # stats = drone_ops.get_drone_status(tello)
    # print(stats)




