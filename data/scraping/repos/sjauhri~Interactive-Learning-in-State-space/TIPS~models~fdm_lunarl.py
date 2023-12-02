"""
Model imported from OpenAI Gym environements:

LunarLander: Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
"""

import math
import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

FPS    = 50
SCALE = 30.0
LANDER_POLY =[
    (-14,+17), (-17,0), (-17,-10),
    (+17,-10), (+17,0), (+14,+17)
    ]
MAIN_ENGINE_POWER  = 13.0
SIDE_ENGINE_POWER  =  0.6
SIDE_ENGINE_AWAY   = 12.0
SIDE_ENGINE_HEIGHT = 14.0
LEG_DOWN = 18

VIEWPORT_W = 600
VIEWPORT_H = 400

def fdm(state, action):        
    #assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))
    # valid actions are 0 (do nothing), 1 (left engine), 2 (main/bottom engine) and 3 (right engine)

    # Create world and lander
    world = Box2D.b2World()
    lander = world.CreateDynamicBody(
    position = ( state[0]*(10.0) + 10.0, state[1]*(40.0/3/2) + ((40.0/3/4) + LEG_DOWN/SCALE) ), # PERFORM SCALING!!!!!Heli = VIEWPORT_H/SCALE/4
    linearVelocity= ( state[2]*5, state[3]*7.5 ),
    angle=state[4]*1.0,
    angularVelocity=state[5]*2.5,
    fixtures = fixtureDef(
        shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in LANDER_POLY ]),
        density=5.0,
        friction=0.1,
        categoryBits=0x0010,
        maskBits=0x001,  # collide only with ground
        restitution=0.0) # 0.99 bouncy
        )

    # Engines
    tip  = (math.sin(lander.angle), math.cos(lander.angle))
    side = (-tip[1], tip[0])
    dispersion = [np.random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

    m_power = 0.0
    if (action==2):
        # Main engine
        m_power = 1.0
        ox =  tip[0]*(4/SCALE + 2*dispersion[0]) + side[0]*dispersion[1]   # 4 is move a bit downwards, +-2 for randomness
        oy = -tip[1]*(4/SCALE + 2*dispersion[0]) - side[1]*dispersion[1]
        impulse_pos = (lander.position[0] + ox, lander.position[1] + oy)
        lander.ApplyLinearImpulse( (-ox*MAIN_ENGINE_POWER*m_power, -oy*MAIN_ENGINE_POWER*m_power), impulse_pos, True)

    s_power = 0.0
    if (action in [1,3]):
        # Orientation engines
        direction = action-2
        s_power = 1.0
        ox =  tip[0]*dispersion[0] + side[0]*(3*dispersion[1]+direction*SIDE_ENGINE_AWAY/SCALE)
        oy = -tip[1]*dispersion[0] - side[1]*(3*dispersion[1]+direction*SIDE_ENGINE_AWAY/SCALE)
        impulse_pos = (lander.position[0] + ox - tip[0]*17/SCALE, lander.position[1] + oy + tip[1]*SIDE_ENGINE_HEIGHT/SCALE)        
        lander.ApplyLinearImpulse( (-ox*SIDE_ENGINE_POWER*s_power, -oy*SIDE_ENGINE_POWER*s_power), impulse_pos, True)

    world.Step(1.0/FPS, 6*30, 2*30)

    pos = lander.position
    vel = lander.linearVelocity
    state = [
        (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
        (pos.y - ((VIEWPORT_H/SCALE/4)+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
        vel.x*(VIEWPORT_W/SCALE/2)/FPS,
        vel.y*(VIEWPORT_H/SCALE/2)/FPS,
        lander.angle,
        20.0*lander.angularVelocity/FPS,
        0.0,
        0.0
        ]    

    return np.array(state, dtype=np.float32)

def fdm_cont(state, action):

    # valid actions are 0,0 to +-1,+-1
    action = np.clip(action, -1, +1).astype(np.float32)

    # Create world and lander
    world = Box2D.b2World()
    lander = world.CreateDynamicBody(
    position = ( state[0]*(10.0) + 10.0, state[1]*(40.0/3/2) + ((40.0/3/4) + LEG_DOWN/SCALE) ), # PERFORM SCALING!!!!!Heli = VIEWPORT_H/SCALE/4
    linearVelocity= ( state[2]*5, state[3]*7.5 ),
    angle=state[4]*1.0,
    angularVelocity=state[5]*2.5,
    fixtures = fixtureDef(
        shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in LANDER_POLY ]),
        density=5.0,
        friction=0.1,
        categoryBits=0x0010,
        maskBits=0x001,  # collide only with ground
        restitution=0.0) # 0.99 bouncy
        )

    # Engines
    tip  = (math.sin(lander.angle), math.cos(lander.angle))
    side = (-tip[1], tip[0])
    dispersion = [np.random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

    m_power = 0.0
    if (action[0] > 0.0):
        # Main engine
        m_power = (np.clip(action[0], 0.0,1.0) + 1.0)*0.5   # 0.5..1.0
        assert m_power>=0.5 and m_power <= 1.0
        ox =  tip[0]*(4/SCALE + 2*dispersion[0]) + side[0]*dispersion[1]   # 4 is move a bit downwards, +-2 for randomness
        oy = -tip[1]*(4/SCALE + 2*dispersion[0]) - side[1]*dispersion[1]
        impulse_pos = (lander.position[0] + ox, lander.position[1] + oy)
        lander.ApplyLinearImpulse( (-ox*MAIN_ENGINE_POWER*m_power, -oy*MAIN_ENGINE_POWER*m_power), impulse_pos, True)

    s_power = 0.0
    if (np.abs(action[1]) > 0.5):
        # Orientation engines
        direction = np.sign(action[1])
        s_power = np.clip(np.abs(action[1]), 0.5,1.0)
        assert s_power>=0.5 and s_power <= 1.0        
        ox =  tip[0]*dispersion[0] + side[0]*(3*dispersion[1]+direction*SIDE_ENGINE_AWAY/SCALE)
        oy = -tip[1]*dispersion[0] - side[1]*(3*dispersion[1]+direction*SIDE_ENGINE_AWAY/SCALE)
        impulse_pos = (lander.position[0] + ox - tip[0]*17/SCALE, lander.position[1] + oy + tip[1]*SIDE_ENGINE_HEIGHT/SCALE)
        lander.ApplyLinearImpulse( (-ox*SIDE_ENGINE_POWER*s_power, -oy*SIDE_ENGINE_POWER*s_power), impulse_pos, True)

    world.Step(1.0/FPS, 6*30, 2*30)

    pos = lander.position
    vel = lander.linearVelocity
    state = [
        (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
        (pos.y - ((VIEWPORT_H/SCALE/4)+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
        vel.x*(VIEWPORT_W/SCALE/2)/FPS,
        vel.y*(VIEWPORT_H/SCALE/2)/FPS,
        lander.angle,
        20.0*lander.angularVelocity/FPS,
        0.0,
        0.0
        ]    

    return np.array(state, dtype=np.float32)


# Test script
# python
# from models.fdm_lunarl import *
# import gym
# env = gym.make('LunarLander-v2')
# st = env.reset()
# st1 = fdm(st, 2)
# st2 = env.step(2)
# st1
# st2
