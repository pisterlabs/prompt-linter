"""
Serves as the primary controller for setting up a device for bluetooth communication with
neighboring devices.

This file needs to be tailored towards your device specific mac address
"""
import csv
import json
import os
import random
import serial
import subprocess
import sys


from guidance.bluetoothctl import Bluetoothctl
from guidance.device import Device
from guidance.monitor import Monitor
from guidance.monitor import *

from bluetooth import *
from time import sleep
from time import time

MOTOR_PIN = 27
BLUETOOTH_PORT = 1
BLUETOOTH_PORT2 = 2
PAYLOAD = 1024
END_TRANSMISSION = b""
PI_ZERO_ADDRESS1 = "B8:27:EB:2D:D7:36"
PI_ZERO_ADDRESS2 = "B8:27:EB:D2:45:EF"
QUERY_TIME_DELTA = 1 # seconds
API_IS_WORKING = True 
PATH_TO_FIFO = "/home/pi/Development/guidance/log_fifo"
PACKET_SIZE = 64
RIGHT_OPTIONS = (
        "turn-slight-right", "turn-sharp-right", "uturn-right",
        "turn-right", "ramp-right", "fork-right", "roundabout-right",
        )


def get_direction(sock):
    """Queries API every QUERY_TIME_DELTA seconds"""
    if not sock:
        with open("./data/dummy_travel_data.csv") as directions:
            for direction in csv.reader(directions):
                # yield direction
                return direction
    else:
        ser = serial.Serial("/dev/ttyUSB0") # default baud rate 9600
        print("Serial Port Initialized...")

        incoming_data = ser.read(1)
        while chr(incoming_data[-1]) != "{":
            print("Incoming Data: ", incoming_data)
            incoming_data += ser.read(1)

        payload_size = incoming_data[:-1]
        print("Payload Size: {}".format(payload_size))
        # payload_size = int(ser.read(4))
        payload_size = int(payload_size)
        primary_chunk_size = payload_size // PACKET_SIZE
        remaining_chunk_size = payload_size - primary_chunk_size - 1

        data = bytes(chr(incoming_data[-1]), "utf-8")
        data += ser.read(primary_chunk_size + remaining_chunk_size)
        print("\nData: {}\n".format(data))

        try:
            data_as_dict = json.loads( data.decode("utf-8") )
        except:
            print("Trouble parsing string as JSON.")

        stepsArray = []
        remaining_distance = 9999
        try:
            stepsArray = data_as_dict["routes"][0]["legs"][0]["steps"]
            remaining_distance = data_as_dict["routes"][0]["legs"][0]["distance"]["text"]
            
            metric = remaining_distance[-2:]
            print("Metric: ", metric)
            s = remaining_distance[: len(remaining_distance) - 2]
            print("Value to convert and type: {} - {}".format(s, type(s)))
            if metric == "mi":
                print("Inside miles...")
                remaining_distance = float(remaining_distance[: len(remaining_distance) - 2] ) * 5280
            else:
                print("Here...")
                remaining_distance = float(remaining_distance[: len(remaining_distance) - 2] )
        except:
            print("Encountered error searching for keys.")

        if remaining_distance < 100:
            ser.close()
            return ["A", "-1"]

        distance = 0
        direction = ""
        if not isinstance(stepsArray, list):
            stepsArray = [stepsArray] # convert to iterable
        
        for step in stepsArray:
            dist = step["distance"]["text"]
            distance += int(dist[: len(dist) - 2 ]) # expecting distance in ft
            if "maneuver" in step:
                direction = "R" if any(right_word in step["maneuver"] for right_word in RIGHT_OPTIONS) else "L" 
                break
        ser.close()
        return [direction, str(distance)]


def get_recipient(direction):
    """Returns the address of the motor to receive some signal.
    
    :param: direction - should be a single character indicating left or right motor
    """
    return PI_ZERO_ADDRESS1 if direction.upper() == "L" else PI_ZERO_ADDRESS2


def process_data(data):
    """When the data is received from the iPhone, process it before sending to pi zeros."""
    if isinstance(data, list):
        # Get data from dummy data source
        return bytes(" ".join(data), "utf-8")
    else:
        # Data is coming in from some api
        # Do some processing
        return data


if __name__ == "__main__":
    btctl = Bluetoothctl()
    device = Device(btctl.get_address(), BLUETOOTH_PORT)
    device2 = Device(btctl.get_address(), BLUETOOTH_PORT2)
    direction = ""
        
    monitor = Monitor()
    monitor.update_screen()

    while device.is_active():
        try:
        # Listen for data
            now = time()
            data = get_direction(True)

            # Process data
            print("Data from api: {}".format(data))
            direction, distance = process_data(data).split(b" ")
            direction = direction.decode("utf-8")
            # direction = random.choice(["L", "R"]) # delete after
            distance = distance.decode("utf-8")
            monitor.set_direction(direction)
            monitor.set_distance(distance)
            monitor.update_screen()
            device.active = monitor.check_screen()

            # Send data
            recipient = get_recipient(direction)
            # print("Here...")
            if not device.is_active() or direction == "A":
                device.active = False
                device.connect(PI_ZERO_ADDRESS1)
                device.send(distance)
                device.close_connection_to_peer()
                device2.connect(PI_ZERO_ADDRESS2)
                device2.send(distance)
                device2.close_connection_to_peer()
            else:
                if recipient == PI_ZERO_ADDRESS1:
                    device.connect(recipient)
                    device.send(distance)
                    device.close_connection_to_peer()
                else:
                    device2.connect(recipient)
                    device2.send(distance)
                    device2.close_connection_to_peer()
            end = time()
            print("Time Delta: {}".format(end - now))
        except KeyboardInterrupt:
            sys.exit(0)
        except:
          # Send the error message to the TFT and retry
            print("There was an error...")
