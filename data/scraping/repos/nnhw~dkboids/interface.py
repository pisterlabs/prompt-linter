from __future__ import print_function
from dronekit import VehicleMode
import argparse
import cmd
import time
from threading import Thread

# local import
import connection
import guidance

# fleet configuration
boids_number = 10


# argument configuration
parser = argparse.ArgumentParser(
    description='DroneKit experiments.')
parser.add_argument('--master',
                    help="vehicle connection target string. If not specified, script connects to 127.0.0.1:14551 by default.")
parser.add_argument('--baud',
                    help="baudrate of the serial connections. Default is 115200.")
parser.add_argument('--safety',
                    help="on/off - disable or enable safety checks. Default is on")
parser.add_argument('--id',
                    help="mavlink system ID of this instance")

args = parser.parse_args()

if not args.id:
    id = 255
else:
    id = int(args.id)


# thread configuration
def start_data_flow_out():
    data_flow_out_thread = Thread(
        target=connection.data_flow_handler_out, args=(
            vehicle, base_update_rate_hz, connection_buddy))
    data_flow_out_thread.daemon = True
    data_flow_out_thread.start()


def start_data_flow_in():
    data_flow_in_thread = Thread(target=connection.data_flow_handler_in, args=(
        vehicle, base_update_rate_hz*boids_number, connection_buddy))
    data_flow_in_thread.daemon = True
    data_flow_in_thread.start()

# utils


def parse(arg):
    'Convert a series of zero or more numbers to an argument tuple'
    if not arg:
        return (0,)
    else:
        return tuple(arg.split())

# main cmd shell


class command_shell(cmd.Cmd):
    intro = 'Welcome to the dkboids command shell. Type help or ? to list commands.\n'
    prompt = '(command) '

    def do_takeoff(self, arg):
        'takeoff <altitude>; load a program, takeoff and reach target altitude'
        flight_level = int((parse(arg))[0])
        vehicle._flight_level = flight_level
        guidance.takeoff(vehicle, flight_level, args.safety)
        print("Taking off completed")

    def do_print_buddy(self, arg):
        'get buddy <1> or <2> or <3> info'
        print(vehicle.get_buddy(int(parse(arg)[0])))

    def do_follow(self, arg):
        'follow target <id>'
        vehicle.mode = VehicleMode("GUIDED")
        vehicle._follow_target_id = int(parse(arg)[0])

    def do_stop_follow(self, arg):
        'stop following'
        vehicle._follow_target_id = 0

    def do_swarm(self, arg):
        'enable swarming behavior and go to the global point of interest'
        vehicle._swarming = True

    def do_stop_swarm(self, arg):
        'disable swarming behavior'
        vehicle._swarming = False

    def do_set_global_poi(self, arg):
        'set global point of interest'
        connection_buddy.send_data({"id": 200, "counter": 0, "flight_level": 0, "lat": float(
            (parse(arg))[0]), "lon": float((parse(arg))[1]), "alt": float((parse(arg))[2]), "groundspeed": 0})

    def do_bye(self, arg):
        'Exit'
        vehicle.close()
        print('Good Bye')
        return True


if __name__ == "__main__":
    base_update_rate_hz = 1
    vehicle = connection.safe_dk_connect(args.master, args.baud, id)
    connection_buddy = connection.buddy_connection(8000)
    start_data_flow_out()
    start_data_flow_in()
    command_shell().cmdloop()
