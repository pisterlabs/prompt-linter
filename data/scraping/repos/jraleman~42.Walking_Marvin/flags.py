#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    flags.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: jaleman <jaleman@student.42.us.org>        +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2017/08/31 02:24:22 by jaleman           #+#    #+#              #
#    Updated: 2017/08/31 02:24:23 by jaleman          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import sys
import argparse
import datetime

from lib.open_ai_gym import OpenAIGym

def parser():
    """
    Flags parser
    """

    parser = argparse.ArgumentParser(
        description="Python project that uses OpenAI Gym with the environment \
            (provided) Marvin. The goal is to train Marvin to walk, \
            having the training and walking process.",
        epilog="Go ahead and run some flags :)")

    parser.add_argument(
        '-w',
        '--walk',
        action='store_true',
        help='display only the walking process',
        required=False)

    parser.add_argument(
        '-v',
        '--video',
        action='store_true',
        help='saves videos of the walking proccess of the best species \
        from each generation',
        required=False)\

    parser.add_argument(
        '-l',
        '--load',
        type=str,
        default=None,
        metavar='PATH',
        help='load weights for Marvin agent from a file \
        (skip training process if this option is specified)',
        required=False)

    parser.add_argument(
        '-s',
        '--save',
        type=str,
        default=None,
        metavar='PATH',
        help='save weights to a file after running the program',
        required=False)

    parser.add_argument(
        '-n',
        '--name',
        type=str,
        default=None,
        metavar='STR',
        help='the name of the game (enviroment)',
        required=False)

    parser.add_argument(
        '-g',
        '--generation',
        type=int,
        default=0,
        metavar='INT',
        help='number of max generations',
        required=False)

    parser.add_argument(
        '-p',
        '--population',
        type=int,
        default=0,
        metavar='INT',
        help='count of the population between each generation',
        required=False)

    parser.add_argument(
        '-r',
        '--rate',
        type=float,
        default=0.0,
        metavar='FLOAT',
        help='mutation rate (recommended values in the range of decimals)',
        required=False)

    parser.add_argument(
        '-m',
        '--movement',
        type=int,
        default=0,
        metavar='INT',
        help='number of steps (movement) between each species',
        required=False)

    parser.add_argument(
        '-q',
        '--quiet',
        action='store_true',
        help='hide the program\'s log between each episode',
        required=False)

    parser.add_argument(
        '--log',
        action='store_true',
        help='save a log of each generation to a file',
        required=False)

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s ' + "0.1.0",
        help="show program's version number and exit")

    return parser.parse_args()

class MarvinFlags(object):
    """
    Class to setup the arguments given from the command line.
    """
    def __init__(self, name, version):
        self.flags = parser()
        self.name = name
        self.version = version
        self.file_name = None
        self.flag_walk = False
        self.flag_video = False
        self.flag_load = False
        self.flag_save = False
        self.flag_name = False
        self.flag_gen = 0
        self.flag_pop = 0
        self.flag_move = 0
        self.flag_rate = 0.0
        self.flag_log = False
        self.flag_quiet = False
        return None

    # Get methods
    def getFlagWalk(self):
        return self.flag_walk
    def getFlagVideo(self):
        return self.flag_video
    def getFlagLoad(self):
        return self.flag_load
    def getFlagSave(self):
        return self.flag_save
    def getFlagName(self):
        return self.flag_name
    def getFlagGen(self):
        return self.flag_gen
    def getFlagPop(self):
        return self.flag_pop
    def getFlagMove(self):
        return self.flag_move
    def getFlagRate(self):
        return self.flag_rate
    def getFlagLog(self):
        return self.flag_log
    def getFlagQuiet(self):
        return self.flag_quiet

    # Set methods
    def setFlagWalk(self, val):
        self.flag_walk = val
        return None
    def setFlagVideo(self, val):
        self.flag_video = val
        return None
    def setFlagLoad(self, val):
        self.flag_load = val
        return None
    def setFlagSave(self, val):
        self.flag_save = val
        return None
    def setFlagName(self, val):
        self.flag_name = val
        return None
    def setFlagGen(self, val):
        self.flag_gen = val
        return None
    def setFlagPop(self, val):
        self.flag_pop = val
        return None
    def setFlagMove(self, val):
        self.flag_move = val
        return None
    def setFlagRate(self, val):
        self.flag_rate = val
        return None
    def setFlagLog(self, val):
        self.flag_log = val
        return None
    def setFlagQuiet(self, val):
        self.flag_quiet = val
        return None

    def loadWeights(best_neural_nets, steps, ai_gym):
        """
        Renders the weights and display the best generations.
        """

        observation = ai_gym.getObservation()
        for i in range(len(best_neural_nets)):
            totalReward = 0
            for step in range(steps):
                ai_gym.getRender()
                action = best_neural_nets[i].getOutput(observation)
                ai_gym.setAction(action)
                observation, reward, done, info = ai_gym.getAction()
                totalReward += reward
                if done:
                    observation = ai_gym.getObservation()
                    break
        return None

    def saveVideo(self, best_neural_nets, steps, ai_gym):
        """
        Save a series of video of the best bots.
        """

        if self.flag_video == True:
            ai_gym.videoMonitor()
        observation = ai_gym.getObservation()
        for i in range(len(best_neural_nets)):
            totalReward = 0
            for step in range(steps):
                action = best_neural_nets[i].getOutput(observation)
                ai_gym.setAction(action)
                observation, reward, done, info = ai_gym.getAction()
                totalReward += reward
                if done:
                    observation = ai_gym.getObservation()
                    break
        return None

    def createLogFile(self):
        """
        Creates a simple log file, with the time date as its name.
        """

        log_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file = self.name + "_" + log_time + ".log"
        sys.stdout = open(log_file, 'a+')
        print("===================")
        print(self.name)
        print(log_time)
        print("===================\n")
        return None

    def saveLog(self, gen, min_fit, avg_fit, max_fit):
        """
        Prints the generation stats.
        """

        if gen == 0:
            print("Stats will be saved in a log file.")
            self.createLogFile()
        print("Generation  : %5d" % (gen + 1))
        print("Min Fitness : %5.0f" % min_fit)
        print("Avg Fitness : %5.0f" % avg_fit)
        print("Max Fitness : %5.0f" % max_fit)
        print("-------------------\n")
        return None


    def initFlags(self):
        """
        Init flags values
        """

        self.setFlagWalk(self.flags.walk)
        self.setFlagVideo(self.flags.video)
        self.setFlagLoad(self.flags.load)
        self.setFlagSave(self.flags.save)
        self.setFlagName(self.flags.name)
        self.setFlagGen(self.flags.generation)
        self.setFlagPop(self.flags.population)
        self.setFlagMove(self.flags.movement)
        self.setFlagRate(self.flags.rate)
        self.setFlagLog(self.flags.log)
        self.setFlagQuiet(self.flags.quiet)
        return None
