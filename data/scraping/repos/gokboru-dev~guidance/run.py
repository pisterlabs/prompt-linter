###################################################################
### Developer(s): umutzan2, ismailtrm                           ###
### Last Update: 19/07/2022 by umutzan2                          ###
### Notes: running code was writen.                             ###
###                                                             ###
###################################################################

import sys
from pymavlink import mavutil
import cv2
from theScript import guidance
import time

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

master = mavutil.mavlink_connection(  # aracin baglantisi
    '/dev/serial0',
    baud=57600)



def set_rc_channel_pwm(id, pwm=1500):

    if id < 1:
        print("Channel does not exist.")
        return

    if id < 9:  # ardusubla iletisim
        rc_channel_values = [65535 for _ in range(8)]
        rc_channel_values[id - 1] = pwm
        master.mav.rc_channels_override_send(
            master.target_system,
            master.target_component,
            *rc_channel_values)


def colorTricking():
    while True:
        success, video = cap.read()  # calling for a cam.
        width = int(cap.get(3))  # screen has created.
        height = int(cap.get(4))

        # function from theScript is configureted.
        x, y = guidance("red", video, 135, 135, width, height)

        if x == 500.5:
            set_rc_channel_pwm(6, 1450)
            set_rc_channel_pwm(3, 1950)
            set_rc_channel_pwm(5, 1600)

        elif x<6 and x<-6:
            set_rc_channel_pwm(6, 1500)
            set_rc_channel_pwm(3, 1500)
            set_rc_channel_pwm(5, 1500)

            while True:
                success2, video2 = cap2.read()  # calling for a cam.
                width2 = int(cap2.get(3))  # screen has created.
                height2 = int(cap2.get(4))

                x2, y2 = guidance("red", video2, 135, 135, width2, height2)
                set_rc_channel_pwm(5, 1600)
                if y2<16 and y2<-16:
                    print("vardık")
                    set_rc_channel_pwm(5, 1500)
                    set_rc_channel_pwm(3, 1950)
                    time.sleep(7)
                    sys.exit()



        elif x > 0 and x < 200:
            set_rc_channel_pwm(6, 1500)
            set_rc_channel_pwm(3, 1500)
            set_rc_channel_pwm(5, 1500)
            print("nesne sağ tarafta")

            while x > 5:
                print(x)
                set_rc_channel_pwm(6, 1550)
            set_rc_channel_pwm(6, 1500)

        elif x < 0:
            set_rc_channel_pwm(6, 1500)
            set_rc_channel_pwm(5, 1500)
            set_rc_channel_pwm(3, 1500)
            print("nesne sol tarafta")

            while x < -5:
                print(x)
                set_rc_channel_pwm(6, 1450)
            set_rc_channel_pwm(6, 1500)

cap.release()
cv2.destroyAllWindows()


colorTricking()
