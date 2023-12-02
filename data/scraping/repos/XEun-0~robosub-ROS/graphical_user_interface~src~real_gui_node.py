from email import message
import time
from tkinter import *
from tkinter import font
from tkinter import ttk

import cv2
import PIL.Image, PIL.ImageTk
import rospy
from guidance_navigation_control.msg import sensorInfo_actuatorStatus
from guidance_navigation_control.msg import controlCommand
from guidance_navigation_control.msg import task_desiredAction
from computer_vision.msg import all_targets
from computer_vision.msg import class_object
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from graphical_user_interface.msg import PID_K_Values

class myGUI():
    """
    Modifier: Aren Petrossian

    Tkinter robot controller for the Autonomous Robotics Class

    """

    def __init__(self):

        print('\nAutonomous Robotics GUI Controller')

        # setup root window
        self.root = Tk()
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # keep the window a fixed size
        self.root.wm_resizable(0, 0)

        self.root.title("Autonomous Robotics GUI Controller")

        self.subSections_font = font.Font(family='Helvetica', size=10, weight='bold')
        ######### create tk variables #########

        # CV variables
        self.bridge = CvBridge()
        self.fps = StringVar(value='8')

        self.buoyObject = StringVar(value='False')
        self.buoyConfidence = StringVar(value='0 %')
        self.buoy_X_axis = StringVar(value='0')
        self.buoy_Y_axis = StringVar(value='0')
        self.buoy_Z_axis = StringVar(value='0')

        self.baseObject = StringVar(value='False')
        self.baseConfidence = StringVar(value='0 %')
        self.base_X_axis = StringVar(value='0')
        self.base_Y_axis = StringVar(value='0')
        self.base_Z_axis = StringVar(value='0')

        # SensorInfo & ActuatorStatus Variables
        self.FL_thruster_percent = IntVar(value=50)
        self.FL_thruster_pwm = StringVar(value='FL: 1500')
        self.L_thruster_percent = IntVar(value=50)
        self.L_thruster_pwm = StringVar(value='L: 1500')
        self.BL_thruster_percent = IntVar(value=50)
        self.BL_thruster_pwm = StringVar(value='BL: 1500')
        self.FR_thruster_percent = IntVar(value=50)
        self.FR_thruster_pwm = StringVar(value='FR: 1500')
        self.R_thruster_percent = IntVar(value=50)
        self.R_thruster_pwm = StringVar(value='R: 1500')
        self.BR_thruster_percent = IntVar(value=50)
        self.BR_thruster_pwm = StringVar(value='BR: 1500')

        self.sensorYaw = StringVar(value='0')
        self.sensorPitch = StringVar(value='0')
        self.sensorRoll = StringVar(value='0')
        self.sensorDepth = StringVar(value='0')
        self.sensorTemp = StringVar(value='0')
        self.sensorStabilized = StringVar(value='10s ago')
        self.lastStab = time.time()
        
        # Setpoint and State Variables
        self.smachYaw = StringVar(value='0')
        self.smachPitch = StringVar(value='0')
        self.smachRoll = StringVar(value='0')
        self.smachDepth = StringVar(value='0')
        self.smachTask = StringVar(value='False | False')
        self.smachState = StringVar(value='Initialize')
        self.state = 0

        # PID K Values
        self.PIDA_values = PID_K_Values()
        self.yaw_P = StringVar(value='2.0')
        self.yaw_I = StringVar(value='0.0')
        self.yaw_D = StringVar(value='0.1059')
        self.yaw_A = StringVar(value='0.0')

        self.pitch_P = StringVar(value='4.0')
        self.pitch_I = StringVar(value='0.1765')
        self.pitch_D = StringVar(value='0.2657')
        self.pitch_A = StringVar(value='50.0')

        self.roll_P = StringVar(value='2.1176')
        self.roll_I = StringVar(value='0.0')
        self.roll_D = StringVar(value='0.2647')
        self.roll_A = StringVar(value='0.0')

        self.horizontal_P = StringVar(value='0.0')
        self.horizontal_I = StringVar(value='0.0')
        self.horizontal_D = StringVar(value='0.0')
        self.horizontal_A = StringVar(value='0.0')

        self.depth_P = StringVar(value='90.0')
        self.depth_I = StringVar(value='7.0588')
        self.depth_D = StringVar(value='26.4706')
        self.depth_A = StringVar(value='7.2')


        # create content window into which everything else is placed
        self.content = ttk.Frame(self.root, padding=12, height=480, width=640)
        self.content.grid(column=0, row=0, sticky=(N, S, E, W))
        self.content.rowconfigure(0, weight=1)

        self.content.columnconfigure(0, weight=1)
        self.content.columnconfigure(1, weight=1)


        # create the GUI content frames
        self.create_left_frame()
        self.create_center_frame()
        self.create_right_frame()
        self.create_bottom_frame()

        # connect to ROS
        rospy.init_node("GUI")
        rospy.Subscriber('sensorInfo_actuatorStatus', sensorInfo_actuatorStatus, self.sensor_actuator_data)
        rospy.Subscriber('controlCommand', controlCommand, self.gnc_data)
        rospy.Subscriber('task_desiredAction', task_desiredAction, self.smach_data)
        rospy.Subscriber('target', all_targets, self.cv_data)
        rospy.Subscriber('Image', Image, self.cvFrame_data)
        self.GUI_pub = rospy.Publisher('PID_K_Values', PID_K_Values, queue_size=10)
        

        # bind the key presses
        self.root.bind("<KeyPress>", self.keyboard)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.root.eval('tk::PlaceWindow . center')
        self.root.mainloop()

    def on_closing(self):
        """
        Destroy the window
        :return:
        """
        self.root.destroy()
        
    def sensor_actuator_data(self, data):
        self.FL_thruster_percent.set(value=int((data.thruster_values[2] - 1100)/8))
        self.FL_thruster_pwm.set(value='FL: ' + str(data.thruster_values[2]))
        self.L_thruster_percent.set(value=int((data.thruster_values[0] - 1100)/8))
        self.L_thruster_pwm.set(value='L: ' + str(data.thruster_values[0]))
        self.BL_thruster_percent.set(value=int((data.thruster_values[4] - 1100)/8))
        self.BL_thruster_pwm.set(value='BL: ' + str(data.thruster_values[4]))
        self.FR_thruster_percent.set(value=int((data.thruster_values[3] - 1100)/8))
        self.FR_thruster_pwm.set(value='FR: ' + str(data.thruster_values[3]))
        self.R_thruster_percent.set(value=int((data.thruster_values[1] - 1100)/8))
        self.R_thruster_pwm.set(value='R: ' + str(data.thruster_values[1]))
        self.BR_thruster_percent.set(value=int((data.thruster_values[5] - 1100)/8))
        self.BR_thruster_pwm.set(value='BR: ' + str(data.thruster_values[5]))

        self.sensorYaw.set(value=str(data.yaw_current))
        self.sensorPitch.set(value=str(data.pitch_current))
        self.sensorRoll.set(value=str(data.roll_current))
        self.sensorDepth.set(value=str(data.depth_current))
        self.sensorTemp.set(value=str(data.temperature))
        if data.stabilized_time == 0:
            self.sensorStabilized.set(value=str(int(time.time() - self.lastStab)) + "s ago")
        else:
            self.lastStab = time.time()
            self.sensorStabilized.set(value="Stabilized")
    
    def gnc_data(self, data):
        self.smachYaw.set(value=str(data.yaw_set))
        self.smachPitch.set(value=str(data.pitch_set))
        self.smachRoll.set(value=str(data.roll_set))
        self.smachDepth.set(value=str(data.depth_set))

    def smach_data(self, data):
        self.smachTask.set(value=str(data.bumpIntoBuoy) + ' | ' + str(data.surface))
        self.smachState.set(value=data.currentState)
        self.stateChange(data)

    def cv_data(self, data):
        self.fps.set(value=str(round(data.fps, 1)))
        self.buoyObject.set(value='False')
        self.buoyConfidence.set(value='0 %')
        self.buoy_X_axis.set(value='0')
        self.buoy_Y_axis.set(value='0')
        self.buoy_Z_axis.set(value='0')
        for target in data.targets:
            if target.name == "camera-box":
                self.buoyObject.set(value='True')
                self.buoyConfidence.set(value=str(round(target.confidence, 1))+ ' %')
                self.buoy_X_axis.set(value=str(round(target.x_angle, 1)))
                self.buoy_Y_axis.set(value=str(round(target.y_distance, 3)))
                self.buoy_Z_axis.set(value=str(round(target.z_distance, 2)))
            else:
                self.buoyObject.set(value='False')
                self.buoyConfidence.set(value='0 %')
                self.buoy_X_axis.set(value='0')
                self.buoy_Y_axis.set(value='0')
                self.buoy_Z_axis.set(value='0')
            if target.name == "home_base":
                self.baseObject.set(value='True')
                self.baseConfidence.set(value=str(round(target.confidence, 1))+ ' %')
                self.base_X_axis.set(value=str(round(target.x_angle, 1)))
                self.base_Y_axis.set(value=str(round(target.y_distance, 3)))
                self.base_Z_axis.set(value=str(round(target.z_distance, 2)))
            else:
                self.baseObject.set(value='False')
                self.baseConfidence.set(value='0 %')
                self.base_X_axis.set(value='0')
                self.base_Y_axis.set(value='0')
                self.base_Z_axis.set(value='0')

    def cvFrame_data(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.video_stream()
        except CvBridgeError:
            print(CvBridgeError)
    
    def PID_output_data(self):
        self.PIDA_values.yaw_PIDA_kP = float(self.yaw_P.get())
        self.PIDA_values.yaw_PIDA_kI = float(self.yaw_I.get())
        self.PIDA_values.yaw_PIDA_kD = float(self.yaw_D.get())
        self.PIDA_values.yaw_PIDA_kA = float(self.yaw_A.get())

        self.PIDA_values.pitch_PIDA_kP = float(self.pitch_P.get())
        self.PIDA_values.pitch_PIDA_kI = float(self.pitch_I.get())
        self.PIDA_values.pitch_PIDA_kD = float(self.pitch_D.get())
        self.PIDA_values.pitch_PIDA_kA = float(self.pitch_A.get())

        self.PIDA_values.roll_PIDA_kP = float(self.roll_P.get())
        self.PIDA_values.roll_PIDA_kI = float(self.roll_I.get())
        self.PIDA_values.roll_PIDA_kD = float(self.roll_D.get())
        self.PIDA_values.roll_PIDA_kA = float(self.roll_A.get())

        self.PIDA_values.horizontal_PIDA_kP = float(self.horizontal_P.get())
        self.PIDA_values.horizontal_PIDA_kI = float(self.horizontal_I.get())
        self.PIDA_values.horizontal_PIDA_kD = float(self.horizontal_D.get())
        self.PIDA_values.horizontal_PIDA_kA = float(self.horizontal_A.get())

        self.PIDA_values.depth_PIDA_kP = float(self.depth_P.get())
        self.PIDA_values.depth_PIDA_kI = float(self.depth_I.get())
        self.PIDA_values.depth_PIDA_kD = float(self.depth_D.get())
        self.PIDA_values.depth_PIDA_kA = float(self.depth_A.get())
        self.GUI_pub.publish(self.PIDA_values)


    def create_left_frame(self):
        """
        Create the left panel of the GUI
        """
        style = ttk.Style()
        style.configure("BW.TLabel", foreground="black", background="white")

        # create the left frame
        left_frame = ttk.Labelframe(self.content, borderwidth=5, text="Computer Vision")
        left_frame.grid(column=0, row=0, sticky=(N, S, E, W))

        #encoder_section_label = ttk.Label(left_frame, text="Camera Feed:", font=left_underline_font)
        #encoder_section_label.grid(column=0, row=0, sticky=(W, E), padx=(10, 1), pady=(0, 5))

        #camera frame section
        #self.cap = cv2.VideoCapture(0)
        self.videoBox = Label(left_frame)
        self.videoBox.grid(column=0, row=1, sticky=W, padx=(5, 5), pady=(1, 5))
        #self.video_stream()

        fps_label = ttk.Label(left_frame, text="FPS:")
        fps_label.grid(column=0, row=2, sticky=W, padx=(250, 5), pady=(1, 5))
        fps_value = ttk.Label(left_frame, width=8, textvariable=self.fps,
                                 relief="sunken", style="BW.TLabel")
        fps_value.grid(column=0, row=2, sticky=W, pady=(1, 5), padx=(310, 20))

        separator = ttk.Separator(left_frame, orient=HORIZONTAL)
        separator.grid(column=0, row=9, sticky=(E, W))

        #buoy object section
        buoySection_label = ttk.Label(left_frame, text="Buoy Object Data:", font=self.subSections_font)
        buoySection_label.grid(column=0, row=10, sticky=(W, E), padx=(10, 1), pady=(5, 10))

        buoyObject_label = ttk.Label(left_frame, text="Object in Frame:")
        buoyObject_label.grid(column=0, row=11, sticky=W, padx=(30, 5), pady=(1, 5))
        buoyObject_value = ttk.Label(left_frame, width=8, textvariable=self.buoyObject,
                                       relief="sunken", style="BW.TLabel")
        buoyObject_value.grid(column=0, row=11, sticky=W, pady=(1, 5), padx=(165, 10))

        buoyConfidence_label = ttk.Label(left_frame, text="Confidence:")
        buoyConfidence_label.grid(column=0, row=12, sticky=W, padx=(30, 5), pady=(1, 5))
        buoyConfidence_value = ttk.Label(left_frame, width=8, textvariable=self.buoyConfidence,
                                            relief="sunken", style="BW.TLabel")
        buoyConfidence_value.grid(column=0, row=12, sticky=W, pady=(1, 5), padx=(165, 20))

        buoy_X_axis_label = ttk.Label(left_frame, text="X Axis (Degrees):")
        buoy_X_axis_label.grid(column=0, row=13, sticky=W, padx=(30, 5), pady=(1, 5))
        buoy_X_axis_value = ttk.Label(left_frame, width=8, textvariable=self.buoy_X_axis,
                                 relief="sunken", style="BW.TLabel")
        buoy_X_axis_value.grid(column=0, row=13, sticky=W, pady=(1, 5), padx=(165, 20))

        buoy_Y_axis_label = ttk.Label(left_frame, text="Y Axis (Meters):")
        buoy_Y_axis_label.grid(column=0, row=14, sticky=W, padx=(30, 5), pady=(1, 5))
        buoy_Y_axis_value = ttk.Label(left_frame, width=8, textvariable=self.buoy_Y_axis,
                                 relief="sunken", style="BW.TLabel")
        buoy_Y_axis_value.grid(column=0, row=14, sticky=W, pady=(1, 5), padx=(165, 20))

        buoy_Z_axis_label = ttk.Label(left_frame, text="Distance (Meters):")
        buoy_Z_axis_label.grid(column=0, row=15, sticky=W, padx=(30, 5), pady=(1, 10))
        buoy_Z_axis_value = ttk.Label(left_frame, width=8, textvariable=self.buoy_Z_axis,
                                 relief="sunken", style="BW.TLabel")
        buoy_Z_axis_value.grid(column=0, row=15, sticky=W, pady=(1, 10), padx=(165, 20))

        #base object section
        baseSection_label = ttk.Label(left_frame, text="Home Object Data:", font=self.subSections_font)
        baseSection_label.grid(column=0, row=10, sticky=(W, E), padx=(310, 1), pady=(5, 10))

        baseObject_label = ttk.Label(left_frame, text="Object in Frame:")
        baseObject_label.grid(column=0, row=11, sticky=W, padx=(330, 5), pady=(1, 5))
        baseObject_value = ttk.Label(left_frame, width=8, textvariable=self.baseObject,
                                       relief="sunken", style="BW.TLabel")
        baseObject_value.grid(column=0, row=11, sticky=W, pady=(1, 5), padx=(465, 10))

        baseConfidence_label = ttk.Label(left_frame, text="Confidence:")
        baseConfidence_label.grid(column=0, row=12, sticky=W, padx=(330, 5), pady=(1, 5))
        baseConfidence_value = ttk.Label(left_frame, width=8, textvariable=self.baseConfidence,
                                            relief="sunken", style="BW.TLabel")
        baseConfidence_value.grid(column=0, row=12, sticky=W, pady=(1, 5), padx=(465, 20))

        base_X_axis_label = ttk.Label(left_frame, text="X Axis (Degrees):")
        base_X_axis_label.grid(column=0, row=13, sticky=W, padx=(330, 5), pady=(1, 5))
        base_X_axis_value = ttk.Label(left_frame, width=8, textvariable=self.base_X_axis,
                                 relief="sunken", style="BW.TLabel")
        base_X_axis_value.grid(column=0, row=13, sticky=W, pady=(1, 5), padx=(465, 20))

        base_Y_axis_label = ttk.Label(left_frame, text="Y Axis (Meters):")
        base_Y_axis_label.grid(column=0, row=14, sticky=W, padx=(330, 5), pady=(1, 5))
        base_Y_axis_value = ttk.Label(left_frame, width=8, textvariable=self.base_Y_axis,
                                 relief="sunken", style="BW.TLabel")
        base_Y_axis_value.grid(column=0, row=14, sticky=W, pady=(1, 5), padx=(465, 20))

        base_Z_axis_label = ttk.Label(left_frame, text="Distance (Meters):")
        base_Z_axis_label.grid(column=0, row=15, sticky=W, padx=(330, 5), pady=(1, 10))
        base_Z_axis_value = ttk.Label(left_frame, width=8, textvariable=self.base_Z_axis,
                                 relief="sunken", style="BW.TLabel")
        base_Z_axis_value.grid(column=0, row=15, sticky=W, pady=(1, 10), padx=(465, 20))


    def create_center_frame(self):
        """
        Create the center panel of the GUI
        """
        # create the center frame
        center_frame = ttk.Labelframe(self.content, borderwidth=5, relief="raised", text="Sensor Info & Actuator Status")
        center_frame.grid(column=1, row=0, sticky=(N, S, E, W))

        sensorSection_label = ttk.Label(center_frame, text="Sensor Data:", font=self.subSections_font)
        sensorSection_label.grid(column=0, row=0, sticky=(W, E), padx=(10, 1), pady=(5, 15))

        sensorYaw_label = ttk.Label(center_frame, text="Yaw (Degrees):")
        sensorYaw_label.grid(column=0, row=1, sticky=W, padx=(35, 5), pady=(5, 25))
        sensorYaw_value = ttk.Label(center_frame, width=8, textvariable=self.sensorYaw,
                                 relief="sunken", style="BW.TLabel")
        sensorYaw_value.grid(column=1, row=1, sticky=W, pady=(5, 25), padx=(5, 15))

        sensorPitch_label = ttk.Label(center_frame, text="Pitch (Degrees):")
        sensorPitch_label.grid(column=0, row=2, sticky=W, padx=(35, 5), pady=(5, 25))
        sensorPitch_value = ttk.Label(center_frame, width=8, textvariable=self.sensorPitch,
                                 relief="sunken", style="BW.TLabel")
        sensorPitch_value.grid(column=1, row=2, sticky=W, pady=(5, 25), padx=(5, 15))

        sensorRoll_label = ttk.Label(center_frame, text="Roll (Degrees):")
        sensorRoll_label.grid(column=0, row=3, sticky=W, padx=(35, 5), pady=(5, 25))
        sensorRoll_value = ttk.Label(center_frame, width=8, textvariable=self.sensorRoll,
                                 relief="sunken", style="BW.TLabel")
        sensorRoll_value.grid(column=1, row=3, sticky=W, pady=(5, 25), padx=(5, 15))


        sensorDepth_label = ttk.Label(center_frame, text="Depth (Meters):")
        sensorDepth_label.grid(column=3, row=1, sticky=W, padx=(20, 5), pady=(5, 25))
        sensorDepth_value = ttk.Label(center_frame, width=8, textvariable=self.sensorDepth,
                                 relief="sunken", style="BW.TLabel")
        sensorDepth_value.grid(column=4, row=1, sticky=W, pady=(5, 25), padx=(15, 15))

        sensorTemp_label = ttk.Label(center_frame, text="Temperature (C):")
        sensorTemp_label.grid(column=3, row=2, sticky=W, padx=(20, 5), pady=(5, 25))
        sensorTemp_value = ttk.Label(center_frame, width=8, textvariable=self.sensorTemp,
                                 relief="sunken", style="BW.TLabel")
        sensorTemp_value.grid(column=4, row=2, sticky=W, pady=(5, 25), padx=(15, 15))

        sensorStabilized_label = ttk.Label(center_frame, text="Last Stabilized:")
        sensorStabilized_label.grid(column=3, row=3, sticky=W, padx=(20, 5), pady=(5, 25))
        sensorStabilized_value = ttk.Label(center_frame, width=8, textvariable=self.sensorStabilized,
                                 relief="sunken", style="BW.TLabel")
        sensorStabilized_value.grid(column=4, row=3, sticky=W, pady=(5, 25), padx=(15, 15))


        separator = ttk.Separator(center_frame, orient=HORIZONTAL)
        separator.grid(column=0, row=4, sticky=(E, W), columnspan=5, pady=(25, 0))

        thrusterSection_label = ttk.Label(center_frame, text="Thruster PWM Data:", font=self.subSections_font)
        thrusterSection_label.grid(column=0, row=5, sticky=(W, E), padx=(10, 1), pady=(10, 15))

        FL_thruster_bar = ttk.Progressbar(center_frame, variable=self.FL_thruster_percent, orient="vertical", mode="determinate", length=150)
        FL_thruster_bar.grid(column=0, row=6, sticky=N)
        FL_thruster_value = ttk.Label(center_frame, width=8, textvariable=self.FL_thruster_pwm,
                                 relief="sunken", style="BW.TLabel")
        FL_thruster_value.grid(column=1, row=6, sticky=W, pady=(25, 25), padx=(25, 25))

        L_thruster_bar = ttk.Progressbar(center_frame, variable=self.L_thruster_percent, orient="horizontal", mode="determinate", length=150)
        L_thruster_bar.grid(column=0, row=7, sticky=W, padx=(15, 15))
        L_thruster_value = ttk.Label(center_frame, width=8, textvariable=self.L_thruster_pwm,
                                 relief="sunken", style="BW.TLabel")
        L_thruster_value.grid(column=1, row=7, sticky=W, pady=(25, 25), padx=(25, 25))

        BL_thruster_bar = ttk.Progressbar(center_frame, variable=self.BL_thruster_percent, orient="vertical", mode="determinate", length=150)
        BL_thruster_bar.grid(column=0, row=8, sticky=N)
        BL_thruster_value = ttk.Label(center_frame, width=8, textvariable=self.BL_thruster_pwm,
                                 relief="sunken", style="BW.TLabel")
        BL_thruster_value.grid(column=1, row=8, sticky=W, pady=(25, 25), padx=(25, 25))

        FR_thruster_bar = ttk.Progressbar(center_frame, variable=self.FR_thruster_percent, orient="vertical", mode="determinate", length=150)
        FR_thruster_bar.grid(column=4, row=6, sticky=N)
        FR_thruster_value = ttk.Label(center_frame, width=8, textvariable=self.FR_thruster_pwm,
                                 relief="sunken", style="BW.TLabel")
        FR_thruster_value.grid(column=3, row=6, sticky=W, pady=(25, 25), padx=(25, 25))

        R_thruster_bar = ttk.Progressbar(center_frame, variable=self.R_thruster_percent, orient="horizontal", mode="determinate", length=150)
        R_thruster_bar.grid(column=4, row=7, sticky=W, padx=(5, 15))
        R_thruster_value = ttk.Label(center_frame, width=8, textvariable=self.R_thruster_pwm,
                                 relief="sunken", style="BW.TLabel")
        R_thruster_value.grid(column=3, row=7, sticky=W, pady=(25, 25), padx=(25, 25))

        BR_thruster_bar = ttk.Progressbar(center_frame, variable=self.BR_thruster_percent, orient="vertical", mode="determinate", length=150)
        BR_thruster_bar.grid(column=4, row=8, sticky=N)
        BR_thruster_value = ttk.Label(center_frame, width=8, textvariable=self.BR_thruster_pwm,
                                 relief="sunken", style="BW.TLabel")
        BR_thruster_value.grid(column=3, row=8, sticky=W, pady=(25, 25), padx=(25, 25))



    def create_right_frame(self):
        """
        Create the right panel of the GUI
        """

        # create the frame
        right_frame = ttk.Labelframe(self.content, borderwidth=5, relief="raised",
                                     text="State Machine")
        right_frame.grid(column=2, row=0, sticky=(N, S, E, W))

        # add the widgets
        smachSection_label = ttk.Label(right_frame, text="Setpoint and State Data:", font=self.subSections_font)
        smachSection_label.grid(column=0, row=0, sticky=(W, E), padx=(10, 1), pady=(5, 15))
        
        smachYaw_label = ttk.Label(right_frame, text="Yaw (Degrees):")
        smachYaw_label.grid(column=0, row=1, sticky=W, padx=(30, 5), pady=(5, 25))
        smachYaw_value = ttk.Label(right_frame, width=5, textvariable=self.smachYaw,
                                 relief="sunken", style="BW.TLabel")
        smachYaw_value.grid(column=1, row=1, sticky=W, pady=(5, 25), padx=(10, 15))

        smachPitch_label = ttk.Label(right_frame, text="Pitch (Degrees):")
        smachPitch_label.grid(column=0, row=2, sticky=W, padx=(30, 5), pady=(5, 25))
        smachPitch_value = ttk.Label(right_frame, width=5, textvariable=self.smachPitch,
                                 relief="sunken", style="BW.TLabel")
        smachPitch_value.grid(column=1, row=2, sticky=W, pady=(5, 25), padx=(10, 15))

        smachRoll_label = ttk.Label(right_frame, text="Roll (Degrees):")
        smachRoll_label.grid(column=0, row=3, sticky=W, padx=(30, 5), pady=(5, 25))
        smachRoll_value = ttk.Label(right_frame, width=5, textvariable=self.smachRoll,
                                 relief="sunken", style="BW.TLabel")
        smachRoll_value.grid(column=1, row=3, sticky=W, pady=(5, 25), padx=(10, 15))


        smachDepth_label = ttk.Label(right_frame, text="Depth (Meters):")
        smachDepth_label.grid(column=3, row=1, sticky=W, padx=(35, 5), pady=(5, 25))
        smachDepth_value = ttk.Label(right_frame, width=5, textvariable=self.smachDepth,
                                 relief="sunken", style="BW.TLabel")
        smachDepth_value.grid(column=4, row=1, sticky=W, pady=(5, 25), padx=(15, 35))

        smachTask_label = ttk.Label(right_frame, text="Bump | Surface:")
        smachTask_label.grid(column=3, row=2, sticky=W, padx=(35, 5), pady=(5, 25))
        smachTask_value = ttk.Label(right_frame, width=11, textvariable=self.smachTask,
                                 relief="sunken", style="BW.TLabel")
        smachTask_value.grid(column=4, row=2, sticky=W, pady=(5, 25), padx=(15, 35))

        smachState_label = ttk.Label(right_frame, text="Current State:")
        smachState_label.grid(column=3, row=3, sticky=W, padx=(35, 5), pady=(5, 25))
        smachState_value = ttk.Label(right_frame, width=11, textvariable=self.smachState,
                                 relief="sunken", style="BW.TLabel")
        smachState_value.grid(column=4, row=3, sticky=W, pady=(5, 25), padx=(15, 35))
        
        self.photoBox = Canvas(right_frame, width=450, height=480)
        self.photoBox.grid(column=0, row=4, sticky=W, padx=(5, 5), pady=(1, 5), columnspan=5)
        self.stateRectangle = self.photoBox.create_rectangle(135,64,283,105, outline="blue", width=3)

        frame = cv2.imread("stateMachine.png")
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = PIL.Image.fromarray(cv2image)
        image_small = img.resize((400,480), PIL.Image.ANTIALIAS)
        imgtk = PIL.ImageTk.PhotoImage(image=image_small)
        self.photoBox.imgtk = imgtk
        self.photoBox.create_image(55,5,anchor=NW, image=imgtk)

        self.stateRectangle = self.photoBox.create_rectangle(135,64,283,105, outline="blue", width=3) #Init Robot
        self.photoBox.grid()


    def create_bottom_frame(self):
        """
        Create the bottom panel of the GUI
        """
        style = ttk.Style()
        style.configure("BW.TLabel", foreground="black", background="white")

        # create the bottom frame
        left_frame = ttk.Labelframe(self.content, borderwidth=5, text="PID K Values")
        left_frame.grid(column=0, row=1, columnspan=3, sticky=(N, S, E, W))

        #buoy object section
        buoySection_label = ttk.Label(left_frame, text="PID Data:", font=self.subSections_font)
        buoySection_label.grid(column=0, row=10, sticky=(W, E), padx=(10, 1), pady=(5, 10))
        PID_P_label = ttk.Label(left_frame, text="P:")
        PID_P_label.grid(column=0, row=10, sticky=W, padx=(150, 5), pady=(1, 5))
        PID_I_label = ttk.Label(left_frame, text="I:")
        PID_I_label.grid(column=0, row=10, sticky=W, padx=(270, 5), pady=(1, 5))
        PID_D_label = ttk.Label(left_frame, text="D:")
        PID_D_label.grid(column=0, row=10, sticky=W, padx=(390, 5), pady=(1, 5))
        PID_A_label = ttk.Label(left_frame, text="A:")
        PID_A_label.grid(column=0, row=10, sticky=W, padx=(510, 5), pady=(1, 5))

        yaw_PID_label = ttk.Label(left_frame, text="Yaw:")
        yaw_PID_label.grid(column=0, row=11, sticky=W, padx=(30, 5), pady=(1, 5))
        yaw_P_entry = Entry(left_frame, width=10, textvariable=self.yaw_P)
        yaw_P_entry.grid(column = 0, row = 11, sticky=W, padx=(150, 5), pady=(1, 5))
        yaw_I_entry = Entry(left_frame, width=10, textvariable=self.yaw_I)
        yaw_I_entry.grid(column = 0, row = 11, sticky=W, padx=(270, 5), pady=(1, 5))
        yaw_D_entry = Entry(left_frame, width=10, textvariable=self.yaw_D)
        yaw_D_entry.grid(column = 0, row = 11, sticky=W, padx=(390, 5), pady=(1, 5))
        yaw_A_entry = Entry(left_frame, width=10, textvariable=self.yaw_A)
        yaw_A_entry.grid(column = 0, row = 11, sticky=W, padx=(510, 5), pady=(1, 5))

        #print(entry_p1.get())
        

        pitch_PID_label = ttk.Label(left_frame, text="Pitch:")
        pitch_PID_label.grid(column=0, row=12, sticky=W, padx=(30, 5), pady=(1, 5))
        pitch_P_entry = Entry(left_frame, width=10, textvariable=self.pitch_P)
        pitch_P_entry.grid(column = 0, row = 12, sticky=W, padx=(150, 5), pady=(1, 5))
        pitch_I_entry = Entry(left_frame, width=10, textvariable=self.pitch_I)
        pitch_I_entry.grid(column = 0, row = 12, sticky=W, padx=(270, 5), pady=(1, 5))
        pitch_D_entry = Entry(left_frame, width=10, textvariable=self.pitch_D)
        pitch_D_entry.grid(column = 0, row = 12, sticky=W, padx=(390, 5), pady=(1, 5))
        pitch_A_entry = Entry(left_frame, width=10, textvariable=self.pitch_A)
        pitch_A_entry.grid(column = 0, row = 12, sticky=W, padx=(510, 5), pady=(1, 5))

        roll_PID_label = ttk.Label(left_frame, text="Roll:")
        roll_PID_label.grid(column=0, row=13, sticky=W, padx=(30, 5), pady=(1, 5))
        roll_P_entry = Entry(left_frame, width=10, textvariable=self.roll_P)
        roll_P_entry.grid(column = 0, row = 13, sticky=W, padx=(150, 5), pady=(1, 5))
        roll_I_entry = Entry(left_frame, width=10, textvariable=self.roll_I)
        roll_I_entry.grid(column = 0, row = 13, sticky=W, padx=(270, 5), pady=(1, 5))
        roll_D_entry = Entry(left_frame, width=10, textvariable=self.roll_D)
        roll_D_entry.grid(column = 0, row = 13, sticky=W, padx=(390, 5), pady=(1, 5))
        roll_A_entry = Entry(left_frame, width=10, textvariable=self.roll_A)
        roll_A_entry.grid(column = 0, row = 13, sticky=W, padx=(510, 5), pady=(1, 5))

        horizontal_PID_label = ttk.Label(left_frame, text="Horizontal:")
        horizontal_PID_label.grid(column=0, row=14, sticky=W, padx=(30, 5), pady=(1, 5))
        horizontal_P_entry = Entry(left_frame, width=10, textvariable=self.horizontal_P)
        horizontal_P_entry.grid(column = 0, row = 14, sticky=W, padx=(150, 5), pady=(1, 5))
        horizontal_I_entry = Entry(left_frame, width=10, textvariable=self.horizontal_I)
        horizontal_I_entry.grid(column = 0, row = 14, sticky=W, padx=(270, 5), pady=(1, 5))
        horizontal_D_entry = Entry(left_frame, width=10, textvariable=self.horizontal_D)
        horizontal_D_entry.grid(column = 0, row = 14, sticky=W, padx=(390, 5), pady=(1, 5))
        horizontal_A_entry = Entry(left_frame, width=10, textvariable=self.horizontal_A)
        horizontal_A_entry.grid(column = 0, row = 14, sticky=W, padx=(510, 5), pady=(1, 5))

        depth_PID_label = ttk.Label(left_frame, text="Depth:")
        depth_PID_label.grid(column=0, row=15, sticky=W, padx=(30, 5), pady=(1, 10))
        depth_P_entry = Entry(left_frame, width=10, textvariable=self.depth_P)
        depth_P_entry.grid(column = 0, row = 15, sticky=W, padx=(150, 5), pady=(1, 5))
        depth_I_entry = Entry(left_frame, width=10, textvariable=self.depth_I)
        depth_I_entry.grid(column = 0, row = 15, sticky=W, padx=(270, 5), pady=(1, 5))
        depth_D_entry = Entry(left_frame, width=10, textvariable=self.depth_D)
        depth_D_entry.grid(column = 0, row = 15, sticky=W, padx=(390, 5), pady=(1, 5))
        depth_A_entry = Entry(left_frame, width=10, textvariable=self.depth_A)
        depth_A_entry.grid(column = 0, row = 15, sticky=W, padx=(510, 5), pady=(1, 5))

        PID_output_button = Button(left_frame, text="Send Values", command=self.PID_output_data)
        PID_output_button.grid(column = 0, row = 13, sticky=W, padx=(700, 5), pady=(1, 5))



    def keyboard(self, event):
        """
        Key press/release event distributor
        """

        if event.keysym == "q":
            self.on_closing()
        #elif event.keysym == "r":
        #    self.stateChange()
        else:
            pass

    
    def video_stream(self):
        #_, frame = self.cap.read()
        cv2image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGBA)
        img = PIL.Image.fromarray(cv2image)
        imgtk = PIL.ImageTk.PhotoImage(image=img)
        self.videoBox.imgtk = imgtk
        self.videoBox.configure(image=imgtk)
        #self.videoBox.after(30, self.video_stream) 


    def stateChange(self, state):
        try:
            self.photoBox.delete(self.lastRectangle)
        except:
            pass
        self.lastRectangle = self.photoBox.create_rectangle(self.photoBox.coords(self.stateRectangle), outline="red", width=2)
        self.photoBox.delete(self.stateRectangle) # 140,64,288,105  +148, 41

        if state.currentState == 'Search: Buoy':
            self.stateRectangle = self.photoBox.create_rectangle(216,128,364,169, outline="blue", width=3)
        elif state.currentState == 'Center: Buoy':
            self.stateRectangle = self.photoBox.create_rectangle(216,192,364,233, outline="blue", width=3)
        elif state.currentState == 'Lost: Buoy':
            self.stateRectangle = self.photoBox.create_rectangle(315,256,425,297, outline="blue", width=3)
        elif state.currentState == 'Bump: Buoy':
            self.stateRectangle = self.photoBox.create_rectangle(163,256,305,297, outline="blue", width=3)
        elif state.currentState == 'Search: Base':
            self.stateRectangle = self.photoBox.create_rectangle(202,319,340,361, outline="blue", width=3)
        elif state.currentState == 'Surface':
            self.stateRectangle = self.photoBox.create_rectangle(219,381,323,425, outline="blue", width=3)
        else:
            self.stateRectangle = self.photoBox.create_rectangle(130,8,288,35, outline="blue", width=3)

        self.photoBox.grid()

                 
if __name__ == "__main__":

    try:
        myGUI()
    except KeyboardInterrupt:
        exit()
