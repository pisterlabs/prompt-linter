"""
Serves as the primary controller for setting up a device for bluetooth communication with
neighboring devices.

This file needs to be tailored
"""
import sys


from guidance.bluetoothctl import Bluetoothctl
from guidance.device import Device
from guidance.motor import Motor


MOTOR_PIN = 27
BLUETOOTH_PORT = 1
PAYLOAD = 1024
END_TRANSMISSION = b"-1"
SLEEP_TIME_DELTA = 5


def signal_handler(sig, frame):
    print("You pressed Ctrl+C")
    sys.exit(0)


if __name__ == "__main__":
    btctl = Bluetoothctl()
    device = Device(btctl.get_address(), BLUETOOTH_PORT)
    motor = Motor(MOTOR_PIN, SLEEP_TIME_DELTA)
    
    while device.is_active():
        try:
            print("Waiting for connection...")
            # Listen for data
            client_sock, client_info = device.accept()
            data = client_sock.recv(PAYLOAD)
            client_sock.close()

            # Translate data to motor command
            distance = int(data)
            print("Data: {}".format(distance))
            if distance < 0:
                device.active = False
                motor.stop_vibrating()
                motor.stop()
            else:
                motor.vibrate(distance)

        except KeyboardInterrupt:
            print("Ending program...")
            sys.exit(0)
        except:
            print("Something bad happened. Trying again.")
