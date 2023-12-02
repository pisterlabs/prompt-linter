
"""Based on: https://gist.github.com/xjcl/8ce64008710128f3a076
    Modified by PedroLopes and ShanYuanTeng for Intro to HCI class but credit remains with author

    HOW TO RUN HOST LOCALLY:
    > python3 pong-audio.py host

    HOW TO RUN HOST FOR CONNECTION:
    > python3 pong-audio.py host --host_ip 127.0.0.1

    HOW TO PLAY ON HOST VISUALLY: 
    Play like a regular pong:
    Player 1 controls the left paddle: UP (W) DOWN (S)
    Player 2 controls the right paddle: UP (O) DOWN (L)

    HOW TO CONNECT TO HOST AS PLAYER 1
    > python3 pong-audio.py player --host_ip 127.0.0.1 --host_port 5005 --player_ip 127.0.0.1 --player_port 5007

    HOW TO CONNECT TO HOST AS PLAYER 2
    > python3 pong-audio.py player --host_ip 127.0.0.1 --host_port 5006 --player_ip 127.0.0.1 --player_port 5008

    about IP and ports: 127.0.0.1 means your own computer, change it to play across computer under the same network. port numbers are picked to avoid conflits.

    CODE YOUR AUDIO CONTROL FOR PLAYER!
    
    p.s.: this needs 10x10 image in the same directory: "white_square.png"."""

import time
import math
import random
import pyglet
import sys
from playsound import playsound
import argparse
from pyo import *
import threading
import json

from pythonosc import osc_server
from pythonosc import dispatcher
from pythonosc import udp_client

mode = ''
debug = False
quit = False

host_ip = "127.0.0.1"
host_port_1 = 5005 # you are player 1 if you talk to this port
host_port_2 = 5006
player_1_ip = "127.0.0.1"
player_2_ip = "127.0.0.1"
player_1_port = 5007
player_2_port = 5008

player_ip = "127.0.0.1"
player_port = 0
host_port = 0

paddle_1 = 225
paddle_2 = 225
paddle_1_direction = 0
paddle_2_direction = 0
snd_left = None
snd_right = None

# store how many powerups each player has
p1_activated = 0
p2_activated = 0
last_power_up = time.time()
power_up_duration = 10
power_up_type = 0
speech_recognition_active = True
latest_voice_command = None

level = 1
game_start = 0


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Program description')
    parser.add_argument('mode', help='host, player (ip & port required)')
    parser.add_argument('--host_ip', type=str, required=False)
    parser.add_argument('--host_port', type=int, required=False)
    parser.add_argument('--player_ip', type=str, required=False)
    parser.add_argument('--player_port', type=int, required=False)
    parser.add_argument('--debug', action='store_true', help='show debug info')
    args = parser.parse_args()
    print("> run as " + args.mode)
    mode = args.mode
    if (args.host_ip):
        host_ip = args.host_ip
    if (args.host_port):
        host_port = args.host_port
    if (args.player_ip):
        player_ip = args.player_ip
    if (args.player_port):
        player_port = args.player_port
    if (args.debug):
        debug = True

# Host
# -------------------------------------#
# used to send messages to players (game state etc)
client_1 = None
client_2 = None

# functions receiving messages from players (game control etc)
def on_receive_game_level(address, args, l):
    global level
    level = l
    if (client_1 != None):
        client_1.send_message("/level", l)
    if (client_2 != None):
        client_2.send_message("/level", l)

def on_receive_game_start(address, args, g):
    global game_start, pause
    game_start = g

def on_receive_paddle_1(address, args, paddle):
    global paddle_1
    print(paddle_1)
    paddle_1 = paddle

def on_receive_movepaddle_1(address, args, direction):
    global paddle_1, paddle_1_direction
    paddle_1_direction = direction

def on_receive_movepaddle_2(address, args, direction):
    global paddle_2, paddle_2_direction
    paddle_2_direction = direction

def on_receive_connection_1(address, args, ip):
    global client_1
    global player_1_ip
    player_1_ip = ip
    client_1 = udp_client.SimpleUDPClient(player_1_ip, player_1_port)
    print("> player 1 connected: " + ip)

def on_receive_paddle_2(address, args, paddle):
    global paddle_2
    paddle_2 = paddle

def on_receive_quit(address, args, b):
    global quit
    print("Quitting...")
    quit = True
    server_1.shutdown()
    server_2.shutdown()
    microphone_thread.join()
    speech_thread.join()
    sound_thread.join()

def on_receive_connection_2(address, args, ip):
    global client_2
    global player_2_ip
    player_2_ip = ip
    client_2 = udp_client.SimpleUDPClient(player_2_ip, player_2_port)
    print("> player 2 connected: " + ip)

def on_receive_bigpaddle_1(address, args, b):
    global p1_activated
    global last_power_up
    if (power_up_type == 3):
        p1_activated = 1
        last_power_up = time.time()
        if (client_1 != None):
            client_1.send_message("/p1bigpaddle", 0)
        if (client_2 != None):
            client_2.send_message("/p1bigpaddle", 0)

def on_receive_bigpaddle_2(address, args, b):
    global p2_activated
    global last_power_up
    if (power_up_type == 4):
        p2_activated = 1
        last_power_up = time.time()
        if (client_1 != None):
            client_1.send_message("/p2bigpaddle", 0)
        if (client_2 != None):
            client_2.send_message("/p2bigpaddle", 0)

dispatcher_1 = dispatcher.Dispatcher()
dispatcher_1.map("/p1", on_receive_paddle_1, "p")
dispatcher_1.map("/l", on_receive_game_level, "l")
dispatcher_1.map("/g", on_receive_game_start, "g")
dispatcher_1.map("/c", on_receive_connection_1, "c")
dispatcher_1.map("/b", on_receive_bigpaddle_1, "b")
dispatcher_1.map("/mp", on_receive_movepaddle_1, "mp")
dispatcher_1.map("/q", on_receive_quit, "q")

dispatcher_2 = dispatcher.Dispatcher()
dispatcher_2.map("/p2", on_receive_paddle_2, "p")
dispatcher_2.map("/l", on_receive_game_level, "l")
dispatcher_2.map("/g", on_receive_game_start, "g")
dispatcher_2.map("/c", on_receive_connection_2, "c")
dispatcher_2.map("/b", on_receive_bigpaddle_2, "b")
dispatcher_2.map("/mp", on_receive_movepaddle_2, "mp")
dispatcher_2.map("/q", on_receive_quit, "q")
# -------------------------------------#

# Player
# TODO: add audio output here so that you can play the game eyes-free
# -------------------------------------#
#play some fun sounds?
s = Server().boot()
s.start()

def hit():
    playsound('hit.wav', False)

def bounce():
    playsound('bounce.wav', False)

def paddle_limit():
    playsound('limit.wav', False)

def sound_manager():
    global game_start, snd_left_1, snd_right_1, snd_left_2, snd_right_2, instructions
    if game_start == 1:
        if client_1 is not None:
            client_1.send_message('/play_continuous_sound', 0)
            client_1.send_message('/adjust_volume', volume_1)
        if client_2 is not None:
            client_2.send_message('/play_continuous_sound',0)
            client_2.send_message('/adjust_volume', volume_2)
    elif game_start == 2:
        if client_1 is not None:
            client_1.send_message('/play_instructions', 0)
        if client_2 is not None:
            client_2.send_message('/play_instructions', 0)
    else:
        if client_1 is not None:
            client_1.send_message('/stop_all_sounds', 0)
        if client_2 is not None:
            client_2.send_message('/stop_all_sounds', 0)

def update_and_send_sound(client):
    global freq, pan, volume_1, volume_2
    if mode == 'p1':
        client.send_message('/update_sound_params', [freq, pan, volume_1])
    elif mode == 'p2':
        client.send_message('/update_sound_params', [freq, pan, volume_2])

def calc_sound(ball_x, ball_y, max_x, max_y):
    global freq, volume, pan, snd_left, snd_right, volume_1, volume_2
    max_freq = 880
    min_freq = 100
    freq = min_freq + ((max_y - ball_y) / max_y) * (max_freq - min_freq)
    pan = ball_x / max_x
    volume_1 = 0.5 + (paddle_1 / max_y) * 0.5
    volume_2 = 0.5 + (paddle_2 / max_y) * 0.5

def play_continuous_sound():
    print("[Sound Manager] Thread started")
    global snd_left, snd_right, game_start, quit, instructions, freq, volume, pan
    instructions = SfPlayer('instructions.wav', loop=False)
    while not quit:
        sound_manager()
        time.sleep(0.1)
    s.stop()

if mode == 'p1':
    host_port = host_port_1
if mode == 'p2':
    host_port = host_port_2

if (mode == 'p1') or (mode == 'p2'):
    client = udp_client.SimpleUDPClient(host_ip, host_port)
    print("> connected to server at "+host_ip+":"+str(host_port))

# functions receiving messages from host
def on_receive_ball(address, *args):
    # print("> ball position: (" + str(args[0]) + ", " + str(args[1]) + ")")
    pass

def on_receive_paddle(address, *args):
    #print("> paddle position: (" + str(args[0]) + ", " + str(args[1]) + ")")
    pass

def on_receive_hitpaddle(address, *args):
    # example sound
    hit()
    #print("> ball hit at paddle " + str(args[0]) )
    pass

def on_receive_ballout(address, *args):
    #print("> ball went out on left/right side: " + str(args[0]) )
    pass

def on_receive_ballbounce(address, *args):
    # example sound
    bounce()
    #print("> ball bounced on up/down side: " + str(args[0]) )

def on_receive_scores(address, *args):
    #print("> scores now: " + str(args[0]) + " vs. " + str(args[1]))
    pass

def on_receive_level(address, *args):
    #print("> level now: " + str(args[0]))
    pass

def on_receive_powerup(address, *args):
    #print("> powerup now: " + str(args[0]))
    pass
    # 1 - freeze p1
    # 2 - freeze p2
    # 3 - adds a big paddle to p1, not use
    # 4 - adds a big paddle to p2, not use

def on_receive_instructions(address, *args):
    pass

def on_receive_p1_limit(address, *args):
    print("p1's paddle hit the limit of screen")
    paddle_limit()

def on_receive_p2_limit(address, *args):
    print("p2's paddle hit the limit of screen")
    paddle_limit()

def on_receive_p1_bigpaddle(address, *args):
    print("> p1 has a big paddle now")
    # when p1 activates their big paddle

def on_receive_p2_bigpaddle(address, *args):
    print("> p2 has a big paddle now")
    # when p2 activates their big paddle

def update_sound_params(address, *args):
    freq, pan, volume = args
    if snd_left is None:
        snd_left = Sine(freq=freq, mul=volume).out(0)
    else:
        snd_left.freq = freq
        snd_left.mul = volume

    if snd_right is None:
        snd_right = Sine(freq=freq, mul=volume).out(1)
    else:
        snd_right.freq = freq
        snd_right.mul = volume

def stop_continuous(address, *args):
    global snd_left, snd_right
    if snd_left and snd_right:
        snd_left.stop()
        snd_right.stop()

def on_receive_instructions(address, *args):
    global instructions
    instructions.out()

def on_receive_play_continuous_sound(address, *args):
    freq, pan, volume = args
    if snd_left is None:
        snd_left = Sine(freq=freq, mul=volume).out(0)
    else:
        snd_left.freq = freq
        snd_left.mul = volume

    if snd_right is None:
        snd_right = Sine(freq=freq, mul=volume).out(1)
    else:
        snd_right.freq = freq
        snd_right.mul = volume

def on_receive_stop_all_sounds(address, *args):
    global snd_right, snd_left
    if snd_left and snd_right:
        snd_left.stop()
        snd_right.stop()
        instructions.stop()

def on_receive_adjust_volume(address, *args):
    volume = args[0]


dispatcher_player = dispatcher.Dispatcher()
dispatcher_player.map("/ball", on_receive_ball)
dispatcher_player.map("/paddle", on_receive_paddle)
dispatcher_player.map("/ballout", on_receive_ballout)
dispatcher_player.map("/ballbounce", on_receive_ballbounce)
dispatcher_player.map("/hitpaddle", on_receive_hitpaddle)
dispatcher_player.map("/scores", on_receive_scores)
dispatcher_player.map("/level", on_receive_level)
dispatcher_player.map("/p1limit", on_receive_p1_limit)
dispatcher_player.map("/p2limit", on_receive_p2_limit)
dispatcher_player.map("/powerup", on_receive_powerup)
dispatcher_player.map("/p1bigpaddle", on_receive_p1_bigpaddle)
dispatcher_player.map("/p2bigpaddle", on_receive_p2_bigpaddle)
dispatcher_player.map("/p2bigpaddle", on_receive_p2_bigpaddle)
dispatcher_player.map("/instructions", on_receive_instructions)
dispatcher_player.map("/play_continuous_sound", on_receive_play_continuous_sound)
dispatcher_player.map("/stop_all_sounds", on_receive_stop_all_sounds)
dispatcher_player.map("/adjust_volume", on_receive_adjust_volume)
# -------------------------------------#

# Player: speech recognition library
# -------------------------------------#
# threading so that listenting to speech would not block the whole program
# speech recognition (default using google, requiring internet)
import speech_recognition as sr
# -------------------------------------#

# Player: pitch & volume detection
# -------------------------------------#
import aubio
import numpy as num
import pyaudio
import wave
import tempfile
import soundfile as sf
from whisper_mic.whisper_mic import WhisperMic
from typing import Optional
import openai

# PyAudio object.
p = pyaudio.PyAudio()
# Open stream.
stream = p.open(format=pyaudio.paFloat32,
    channels=1, rate=44100, input=True,
    frames_per_buffer=1024)
# Aubio's pitch detection.
pDetection = aubio.pitch("default", 2048,
    2048//2, 44100)
# Set unit.
pDetection.set_unit("Hz")
pDetection.set_silence(-40)
# -------------------------------------#

# keeping score of points:
p1_score = 0
p2_score = 0
selected_device_index = 1
recognizer = sr.Recognizer()

# Player: speech recognition functions using google api
# TODO: you can use this for input, add function like "client.send_message()" to control the host game
# -------------------------------------#
'''print("Available audio input devices:")
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev['maxInputChannels'] > 0:
        print(f"Device index {i}: {dev['name']}")'''

def listen_to_speech():
    global quit, latest_voice_command
    print("[speech recognition] Thread started")
    mic = WhisperMic(model="base", english=True, verbose=False, energy=200, pause=0.5, dynamic_energy=True, save_file=False)
    while not quit:
        try:
            recog_results = mic.listen()
            latest_voice_command = recog_results.lower()
            print(f"[speech recognition] Recognized: {latest_voice_command}")
            speech_processor(latest_voice_command)
        except sr.UnknownValueError:
            print("[speech recognition] Whisper could not understand audio")
        except sr.RequestError as e:
            print(f"[speech recognition] Could not request results from Whisper service; {e}")

def speech_processor(command):
    global quit
    global paddle_1, paddle_2, paddle_1_direction, paddle_2_direction
    number_commands = {'one':675, 'two':600, 'three':525, 'four':450, 'five':375, 'six':300, 'seven':225, 'eight':150, 'nine':75, 'ten':0}
    if 'start' in command or 'play' in command:
        client.send_message('/g', 1)
    if 'pause' in command:
        client.send_message('/g', 0)
    if 'menu' in command:
        client.send_message('/g', 0)
    if 'quit' in command:
        client.send_message('/q', 0)
    if 'level one' in command:
        client.send_message('/l', 1)
    elif 'level two' in command:
        client.send_message('/l', 2)
    elif 'level three' in command:
        client.send_message('/l', 3)
    for num_com, num in number_commands.items():
        if num_com in command:
            if mode == 'p1':
                client.send_message('/p1', int(num))
                break
            elif mode == 'p2':
                client.send_message('/p2', int(num))
                break
    if 'power' in command:
        client.send_message('/b',0)
    if 'up' in command:
        client.send_message('/mp',-1)
    elif 'down' in command:
        client.send_message('/mp', 1)
    elif 'stop' in command:
        client.send_message('/mp', 0)
    if 'instructions' in command or 'help' in command:
        client.send_message('/g',2)

# -------------------------------------#

# Player: pitch & volume detection
# TODO: you can use this for input, add function like "client.send_message()" to control the host game
# -------------------------------------#
def sense_microphone():
    global quit
    global paddle_1, paddle_2, mode
    global debug
    
    data = stream.read(1024,exception_on_overflow=False)
    samples = num.frombuffer(data,
        dtype=aubio.float_type)

        # Compute the pitch of the microphone input
    pitch = pDetection(samples)[0]
        # Compute the energy (volume) of the mic input
    volume = num.sum(samples**2)/len(samples)
        # Format the volume output so that at most
        # it has six decimal numbers.
    volume = "{:.6f}".format(volume)
        # uncomment these lines if you want pitch or volume
    if debug:
        print("pitch "+str(pitch)+" volume "+str(volume))
# -------------------------------------#

# Host game mechanics: no need to change below
class Ball(object):

    def __init__(self):
        self.debug = 0
        self.TO_SIDE = 5
        self.x = 50.0 + self.TO_SIDE
        self.y = float( random.randint(0, 675) )
        self.x_old = self.x  # coordinates in the last frame
        self.y_old = self.y
        self.vec_x = 2**0.5 / 2  # sqrt(2)/2
        self.vec_y = random.choice([-1, 1]) * 2**0.5 / 2

class Player(object):

    def __init__(self, NUMBER, screen_WIDTH=1200):
        """NUMBER must be 0 (left player) or 1 (right player)."""
        self.NUMBER = NUMBER
        self.x = 50.0 + (screen_WIDTH - 100) * NUMBER
        self.y = 50.0
        self.last_movements = [0]*4  # short movement history
                                     # used for bounce calculation
        self.up_key, self.down_key = None, None
        if NUMBER == 0:
            self.up_key = pyglet.window.key.W
            self.down_key = pyglet.window.key.S
        elif NUMBER == 1:
            self.up_key = pyglet.window.key.O
            self.down_key = pyglet.window.key.L

class Model(object):
    """Model of the entire game. Has two players and one ball."""

    def __init__(self, DIMENSIONS=(1200, 675)):
        """DIMENSIONS is a tuple (WIDTH, HEIGHT) of the field."""
        # OBJECTS
        WIDTH = DIMENSIONS[0]
        self.players = [Player(0, WIDTH), Player(1, WIDTH)]
        self.ball = Ball()
        # DATA
        self.pressed_keys = set()  # set has no duplicates
        self.quit_key = pyglet.window.key.Q
        self.p1activate_key = pyglet.window.key.E
        self.p2activate_key = pyglet.window.key.P
        self.menu_key = pyglet.window.key.SPACE
        self.level_1_key = pyglet.window.key._1
        self.level_2_key = pyglet.window.key._2
        self.level_3_key = pyglet.window.key._3
        self.instructions_key = pyglet.window.key.I
        self.speed = 4  # in pixels per frame
        self.ball_speed = self.speed #* 2.5
        self.WIDTH, self.HEIGHT = DIMENSIONS
        # STATE VARS
        self.menu = 0 # 0: menu, 1: game, 2: instructions
        self.level = 1
        self.paused = True
        self.i = 0  # "frame count" for debug
        self.powerup = 0 # (0=none, 1=player_1, 2=player_2)

    def reset_ball(self, who_scored):
        """Place the ball anew on the loser's side."""
        if debug: print(str(who_scored)+" scored. reset.")
        self.ball.y = float( random.randint(0, self.HEIGHT) )
        self.ball.vec_y = random.choice([-1, 1]) * 2**0.5 / 2
        if who_scored == 0:
            self.ball.x = self.WIDTH - 50.0 - self.ball.TO_SIDE
            self.ball.vec_x = - 2**0.5 / 2
        elif who_scored == 1:
            self.ball.x = 50.0 + self.ball.TO_SIDE
            self.ball.vec_x = + 2**0.5 / 2
        elif who_scored == "debug":
            self.ball.x = 70  # in paddle atm -> usage: hold f
            self.ball.y = self.ball.debug
            self.ball.vec_x = -1
            self.ball.vec_y = 0
            self.ball.debug += 0.2
            if self.ball.debug > 100:
                self.ball.debug = 0

    def check_if_oob_top_bottom(self):
        """Called by update_ball to recalc. a ball above/below the screen."""
        # bounces. if -- bounce on top of screen. elif -- bounce on bottom.
        b = self.ball
        if b.y - b.TO_SIDE < 0:
            illegal_movement = 0 - (b.y - b.TO_SIDE)
            b.y = 0 + b.TO_SIDE + illegal_movement
            b.vec_y *= -1
            if (client_1 != None):
                client_1.send_message("/ballbounce", 1)
            if (client_2 != None):
                client_2.send_message("/ballbounce", 1)
        elif b.y + b.TO_SIDE > self.HEIGHT:
            illegal_movement = self.HEIGHT - (b.y + b.TO_SIDE)
            b.y = self.HEIGHT - b.TO_SIDE + illegal_movement
            b.vec_y *= -1
            if (client_1 != None):
                client_1.send_message("/ballbounce", 2)
            if (client_2 != None):
                client_2.send_message("/ballbounce", 2)

    def check_if_oob_sides(self):
        global p2_score, p1_score
        """Called by update_ball to reset a ball left/right of the screen."""
        b = self.ball
        if b.x + b.TO_SIDE < 0:  # leave on left
            self.reset_ball(1)
            p2_score+=1
            if (client_1 != None):
                client_1.send_message("/ballout", 1)
                client_1.send_message("/scores", [p1_score, p2_score])
            if (client_2 != None):
                client_2.send_message("/ballout", 1)
                client_2.send_message("/scores", [p1_score, p2_score])
        elif b.x - b.TO_SIDE > self.WIDTH:  # leave on right
            p1_score+=1
            self.reset_ball(0)
            if (client_1 != None):
                client_1.send_message("/ballout", 2)
                client_1.send_message("/scores", [p1_score, p2_score])
            if (client_2 != None):
                client_2.send_message("/ballout", 2)
                client_2.send_message("/scores", [p1_score, p2_score])

    def check_if_paddled(self): 
        """Called by update_ball to recalc. a ball hit with a player paddle."""
        b = self.ball
        p0, p1 = self.players[0], self.players[1]
        angle = math.acos(b.vec_y)  
        factor = random.randint(5, 15)  
        cross0 = (b.x < p0.x + 2*b.TO_SIDE) and (b.x_old >= p0.x + 2*b.TO_SIDE)
        cross1 = (b.x > p1.x - 2*b.TO_SIDE) and (b.x_old <= p1.x - 2*b.TO_SIDE)
        if p1_activated == 1 and power_up_type == 3:
            bounding_1 = 25 * 6
        else: 
            bounding_1 = 25 * 4
        if cross0 and -bounding_1 < b.y - p0.y < bounding_1:
            hit()
            if (client_1 != None):
                client_1.send_message("/hitpaddle", 1)
            if (client_2 != None):
                client_2.send_message("/hitpaddle", 1)
            if debug: print("hit at "+str(self.i))
            illegal_movement = p0.x + 2*b.TO_SIDE - b.x
            b.x = p0.x + 2*b.TO_SIDE + illegal_movement
            # angle -= sum(p0.last_movements) / factor / self.ball_speed
            b.vec_y = math.cos(angle)
            b.vec_x = (1**2 - b.vec_y**2) ** 0.5
        else: 
            if p2_activated == 1 and power_up_type == 4:
                bounding = 25 * 6
            else: 
                bounding = 25 * 4
            if cross1 and -bounding < b.y - p1.y < bounding:
                hit()
                if (client_1 != None):
                    client_1.send_message("/hitpaddle", 2)
                if (client_2 != None):
                    client_2.send_message("/hitpaddle", 2)
                if debug: print("hit at "+str(self.i))
                illegal_movement = p1.x - 2*b.TO_SIDE - b.x
                b.x = p1.x - 2*b.TO_SIDE + illegal_movement
                # angle -= sum(p1.last_movements) / factor / self.ball_speed
                b.vec_y = math.cos(angle)
                b.vec_x = - (1**2 - b.vec_y**2) ** 0.5

# -------------- Ball position: you can find it here -------
    def update_ball(self):
        """
            Update ball position with post-collision detection.
            I.e. Let the ball move out of bounds and calculate
            where it should have been within bounds.

            When bouncing off a paddle, take player velocity into
            consideration as well. Add a small factor of random too.
        """
        global client_1
        global client_2
        global snd_left_1, snd_right_1
        self.i += 1  # "debug"
        b = self.ball
        b.x_old, b.y_old = b.x, b.y
        b.x += b.vec_x * self.ball_speed 
        b.y += b.vec_y * self.ball_speed
        self.check_if_oob_top_bottom()  # oob: out of bounds
        self.check_if_oob_sides()
        self.check_if_paddled()
        if (client_1 != None):
            client_1.send_message("/ball", [b.x, b.y])
        if (client_2 != None):
            client_2.send_message("/ball", [b.x, b.y])
        if snd_left_1 is not None:
            calc_sound(self.ball.x, self.ball.y, self.WIDTH, self.HEIGHT)
            

    def toggle_menu(self):
        global game_start
        if (self.menu != 0):
            self.menu = 0
            game_start = 0
            self.paused = True
        else:
            self.menu = 1
            game_start = 1
            self.paused = False
    
    def toggle_instructions(self):
        global game_start
        if self.menu != 2:
            self.menu = 2
            game_start = 2
            self.paused = True
        else:
            self.menu = 0
            game_start = 0
            self.paused = False

    def update(self):
        """Work through all pressed keys, update and call update_ball."""
        global paddle_1, paddle_1_direction
        global paddle_2, paddle_2_direction
        global p1_activated
        global p2_activated
        global snd_left, snd_right
        # you can change these to voice input too
        pks = self.pressed_keys
        if quit:
            sys.exit(1)
        if self.quit_key in pks:
            exit(0)
        if self.menu_key in pks:
            self.toggle_menu()
            pks.remove(self.menu_key) # debounce: get rid of quick duplicated presses
        if self.instructions_key in pks:
            self.toggle_instructions()
            pks.remove(self.instructions_key)

        if self.p1activate_key in pks:
            # print("E pressed to send power up on 1")
            if power_up_type == 3:
                p1_activated = 1
                last_power_up = time.time() #pedro added 2023
            # else: 
                #print("... but there's none active for P1")
            pks.remove(self.p1activate_key)
        if self.p2activate_key in pks:
            # print("P pressed to send power up by P2")
            if power_up_type == 4:
                p2_activated = 1
                last_power_up = time.time() #pedro added 2023
            # else: 
                # print("... but there's none active for P2")
            pks.remove(self.p2activate_key)
        if self.level_1_key in pks:
            self.level = 1
            self.ball_speed = self.speed
            pks.remove(self.level_1_key)
        if self.level_2_key in pks:
            self.level = 2
            self.ball_speed = self.speed*2
            pks.remove(self.level_2_key)
        if self.level_3_key in pks:
            self.level = 3
            self.ball_speed = self.speed*3
            pks.remove(self.level_3_key)
        if pyglet.window.key.R in pks and debug:
            self.reset_ball(1)
        if pyglet.window.key.F in pks and debug:
            self.reset_ball("debug")

        if not self.paused:
            p1 = self.players[0]
            p2 = self.players[1]

            if power_up_type == 1:
                pass
            else: 
                if (paddle_1 != 0):
                    p1.y = paddle_1
                    paddle_1 = 0
                if p1.up_key in pks and p1.down_key not in pks: 
                    p1.y -= (1.5*self.speed)
                elif p1.up_key not in pks and p1.down_key in pks: 
                    p1.y += (1.5*self.speed)
            
            if power_up_type == 1:
                pass
            else: 
                if paddle_1_direction == -1:
                    if p1.y - self.speed > 0:
                        p1.y -= self.speed
                    else:
                        paddle_1_direction = 0
                        client_1.send_message("/p1limit", 0)
                if paddle_1_direction == 1:
                    if p1.y + self.speed < 675:
                        p1.y += self.speed
                    else:
                        paddle_1_direction = 0

            if power_up_type == 2:
                pass
            else: 
                if (paddle_2 != 0):
                    p2.y = paddle_2
                    paddle_2 = 0
                if p2.up_key in pks and p2.down_key not in pks:
                    p2.y -= (1.5*self.speed)
                elif p2.up_key not in pks and p2.down_key in pks:
                    p2.y += (1.5*self.speed)
            
            if power_up_type == 1:
                pass
            else: 
                if paddle_2_direction == -1:
                    if p2.y - self.speed > 0:
                        p2.y -= self.speed
                    else:
                        paddle_2_direction = 0
                        client_2.send_message("/p2limit", 0)
                if paddle_2_direction == 1:
                    if p2.y + self.speed < 675:
                        p2.y += self.speed
                    else:
                        paddle_2_direction = 0

            if (client_1 != None):
                client_1.send_message("/paddle", [p1.y, p2.y])
            if (client_2 != None):
                client_2.send_message("/paddle", [p1.y, p2.y])

            self.update_ball()

class Controller(object):

    def __init__(self, model):
        self.m = model

    def on_key_press(self, symbol, modifiers):
        # `a |= b`: mathematical or. add to set a if in set a or b.
        # equivalent to `a = a | b`.
        # p0 holds down both keys => p1 controls break  # PYGLET!? D:
        self.m.pressed_keys |= set([symbol])

    def on_key_release(self, symbol, modifiers):
        if symbol in self.m.pressed_keys:
            self.m.pressed_keys.remove(symbol)

    def update(self):
        self.m.update()


class View(object):

    def __init__(self, window, model):
        self.w = window
        self.m = model
        # ------------------ IMAGES --------------------#
        # "white_square.png" is a 10x10 white image
        lplayer = pyglet.resource.image("white_square.png")
        self.player_spr = pyglet.sprite.Sprite(lplayer)

    def redraw_game(self):
        # ------------------ PLAYERS --------------------#
        TO_SIDE = self.m.ball.TO_SIDE
        idx = 0
        for p in self.m.players:
                idx = idx + 1                       
                self.player_spr.x = p.x//1 - TO_SIDE
                # oh god! pyglet's (0, 0) is bottom right! madness.
                self.player_spr.y = self.w.height - (p.y//1 + TO_SIDE)
                self.player_spr.draw()  # these 3 lines: pretend-paddle
                self.player_spr.y += 2*TO_SIDE; self.player_spr.draw()
                self.player_spr.y += 2*TO_SIDE; self.player_spr.draw()
                self.player_spr.y += 2*TO_SIDE; self.player_spr.draw()
                self.player_spr.y += 2*TO_SIDE; self.player_spr.draw()
                # print ("----")
                # print (p1_activated)
                # print (p2_activated)
                # print(power_up_type)
                if idx == 2 and p2_activated == 1 and power_up_type == 4:
                    self.player_spr.y += 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y += 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y += 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y += 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y += 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y += 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y += 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y += 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y -= 14*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y -= 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y -= 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y -= 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y -= 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y -= 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y -= 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y -= 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y -= 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y -= 2*TO_SIDE; self.player_spr.draw()

# do the same for p1
                if idx == 1 and p1_activated == 1 and power_up_type == 3:
                    self.player_spr.y += 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y += 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y += 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y += 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y += 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y += 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y += 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y -= 14*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y -= 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y -= 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y -= 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y -= 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y -= 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y -= 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y -= 2*TO_SIDE; self.player_spr.draw()
                    self.player_spr.y -= 2*TO_SIDE; self.player_spr.draw()
 
                
        # ------------------ BALL --------------------#
        self.player_spr.x = self.m.ball.x//1 - TO_SIDE
        self.player_spr.y = self.w.height - (self.m.ball.y//1 + TO_SIDE)
        self.player_spr.draw()
        

    def redraw_menu(self):
        global level
        self.m.level = level
        if (level == 1):
            self.m.ball_speed = self.m.speed
        elif (level == 2):
            self.m.ball_speed = self.m.speed*2
        elif (level == 3):
            self.m.ball_speed = self.m.speed*3
        self.start_label = pyglet.text.Label("press space to start", font_name=None, font_size=36, x=self.w.width//2, y=self.w.height//2, anchor_x='center', anchor_y='center')
        self.level_label = pyglet.text.Label("easy | hard | insane", font_name=None, font_size=24, x=self.w.width//2, y=self.w.height//2+100, anchor_x='center', anchor_y='center')
        if (self.m.level == 1):
            self.level_indicator_label = pyglet.text.Label("------", font_name=None, font_size=24, x=self.w.width//2-105, y=self.w.height//2+80, anchor_x='center', anchor_y='center')
        elif (self.m.level == 2):
            self.level_indicator_label = pyglet.text.Label("------", font_name=None, font_size=24, x=self.w.width//2-12, y=self.w.height//2+80, anchor_x='center', anchor_y='center')
        elif (self.m.level == 3):
            self.level_indicator_label = pyglet.text.Label("---------", font_name=None, font_size=24, x=self.w.width//2+92, y=self.w.height//2+80, anchor_x='center', anchor_y='center')
        self.start_label.draw()
        self.level_label.draw()
        self.level_indicator_label.draw()
    
    def redraw_instructions(self):
        window_height = self.w.height
        instruction_y_start = window_height - 60
        self.instructions_label = pyglet.text.Label("Instructions", font_name=None, font_size=36, x=self.w.width//2, y=instruction_y_start, anchor_x='center', anchor_y='center', color=(255, 255, 255, 255))
        self.instructions_label.draw()

        instruction_lines = [
            "Listen to the pitch to find the y-position of the ball. Pay attention to the stereo sound to find the x-position.",
            "You will hear sounds when you hit the ball and when the ball bounces on the walls.",
            "When your paddle is closer to the top of the screen, the ball's frequency will be louder, and vice-versa.",
            "1. To start the game, say 'start' or 'play'",
            "2. To pause the game, say 'pause' or 'menu'",
            "3. To quit the game, say 'quit'",
            "4. To choose a level, say 'Level + [Level Number]'",
            "5. To move your paddle continuously, say 'up' or 'down' then 'stop' when you want to stop moving",
            "6. To move your paddle incrementally, say a number between 1 and 10, 10 being the top",
            "7. To activate a powerup, say 'power'"
        ]

        line_height = 20
        vertical_spacing = 20
        current_y = instruction_y_start - 60

        for line in instruction_lines:
            label = pyglet.text.Label(line, font_name=None, font_size=18, x=self.w.width//2, y=current_y, anchor_x='center', anchor_y='center', color=(255, 255, 255, 255))
            label.draw()
            current_y -= (line_height + vertical_spacing)


class Window(pyglet.window.Window):

    def __init__(self, *args, **kwargs):
        DIM = (1200, 675)  # DIMENSIONS
        super(Window, self).__init__(width=DIM[0], height=DIM[1],
                                     *args, **kwargs)
        # ------------------ MVC --------------------#
        the_window = self
        self.model = Model(DIM)
        self.view2 = View(the_window, self.model)
        self.controller = Controller(self.model)
        # ------------------ CLOCK --------------------#
        fps = 60.0
        pyglet.clock.schedule_interval(self.update, 1.0/fps)
        #pyglet.clock.set_fps_limit(fps)

        self.score_label = pyglet.text.Label(str(p1_score)+':'+str(p2_score), font_name=None, font_size=36, x=self.width//2, y=self.height//2, anchor_x='center', anchor_y='center')
        self.powerup_status_label = pyglet.text.Label("status: ", font_name=None, font_size=16, x=self.width//2, y=self.height//8, anchor_x='center', anchor_y='center')

    def on_key_release(self, symbol, modifiers):
        self.controller.on_key_release(symbol, modifiers)

    def on_key_press(self, symbol, modifiers):
        self.controller.on_key_press(symbol, modifiers)

    def on_close(self):
        pyglet.app.exit()

    def update(self, *args, **kwargs):
        global last_power_up
        global power_up_duration
        global power_up_type
        global p1_activated
        global p2_activated
        global quit
        # make more efficient (save last position, draw black square
        # over that and the new square, don't redraw _entire_ frame.)
        self.clear()
        self.controller.update()
        
        self.model.menu = game_start

        if quit:
            self.close()

        if (game_start == 1):
            self.model.paused = False
        else:
            self.model.paused = True

        if self.model.menu == 2:
            self.view2.redraw_instructions()
        elif self.model.menu == 1:
            self.view2.redraw_game()
            self.score_label.draw()
        else:
            self.view2.redraw_menu()

        if (time.time() > last_power_up + random.randint(20,32)):
            last_power_up = time.time()
            power_up_type = random.randint(1,4)
            # print("new powerup: " + str(power_up_type))
            # 1 - freeze p1
            # 2 - freeze p2
            # 3 - adds a big paddle to p1, not use
            # 4 - adds a big paddle to p2, not use

            if (client_1 != None):
                # fix power up you / oppenent fre
                client_1.send_message("/powerup", power_up_type)
            if (client_2 != None):
                client_2.send_message("/powerup", power_up_type)

        if (power_up_type != 0 and time.time() > last_power_up + power_up_duration):
            # print("reset powerup")
            power_up_type = 0
            p1_activated = 0
            p2_activated = 0
            if (client_1 != None):
                client_1.send_message("/powerup", 0)
            if (client_2 != None):
                client_2.send_message("/powerup", 0)

        self.score_label.text = str(p1_score)+':'+str(p2_score)
        if power_up_type == 1:
            power_up_status_add = " P1 is frozen!"
        elif power_up_type == 2:
            power_up_status_add = " P2 is frozen!"
        elif power_up_type == 3:
            power_up_status_add = " P1 could use big-paddle now!"
        elif power_up_type == 4:
            power_up_status_add = " P2 could use big-paddle now!"
        else:
            power_up_status_add = " no active power ups" 
        self.powerup_status_label.text = "powerup status: " + power_up_status_add 
        self.powerup_status_label.draw()

if mode == 'host':
    # OSC thread
    # -------------------------------------#
    server_1 = osc_server.ThreadingOSCUDPServer((host_ip, host_port_1), dispatcher_1)
    server_1_thread = threading.Thread(target=server_1.serve_forever)
    server_1_thread.daemon = True
    server_1_thread.start()
    server_2 = osc_server.ThreadingOSCUDPServer((host_ip, host_port_2), dispatcher_2)
    server_2_thread = threading.Thread(target=server_2.serve_forever)
    server_2_thread.daemon = True
    server_2_thread.start()
    print("> server opens at ip: "+host_ip)
    print("> instruction: player 1 connects to "+str(host_port_1) + ", listen at "+str(player_1_port))
    print("> instruction: player 2 connects to "+str(host_port_2) + ", listen at "+str(player_2_port))
    # -------------------------------------#

# Host: pygame starts
if mode == 'host':
    window = Window()
    sound_thread = threading.Thread(target=play_continuous_sound)
    sound_thread.daemon = True
    sound_thread.start()
    pyglet.app.run()


if (mode == 'p1') or (mode == 'p2'):

    microphone_thread = threading.Thread(target=sense_microphone, args=())
    microphone_thread.daemon = True
    microphone_thread.start()

    speech_thread = threading.Thread(target=listen_to_speech, args=())
    speech_thread.daemon = True
    speech_thread.start()

# Player
if mode == 'p1':
    player_port = player_1_port
if mode == 'p2':
    player_port = player_2_port

if (mode == 'p1') or (mode == 'p2'):

    # OSC thread
    # -------------------------------------#
    player_server = osc_server.ThreadingOSCUDPServer((player_ip, player_port), dispatcher_player)
    player_server_thread = threading.Thread(target=player_server.serve_forever)
    player_server_thread.daemon = True
    player_server_thread.start()
    # -------------------------------------#
    client.send_message("/c", player_ip)

    # manual input for debugging
    while True:
      m = input("> send: ")
      cmd = m.split(' ')
      if len(cmd) == 2:
        client.send_message("/"+cmd[0], int(cmd[1]))
      if len(cmd) == 1:
        client.send_message("/"+cmd[0], 0)

      # this is how client send messages to server
      # send paddle position 200 (it should be between 0 - 450):
      # client.send_message('/p', 200)
      # set level to 3:
      # client.send_message('/l', 3)
      # start the game:
      # client.send_message('/g', 1)
      # pause the game:
      # client.send_message('/g', 0)
      # big paddle if received power up:
      # client.send_message('/b', 0)
