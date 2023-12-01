#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy2 Experiment Builder (v1.83.04), Thu Oct  6 14:14:43 2016
If you publish work using this script please cite the relevant PsychoPy publications
  Peirce, JW (2007) PsychoPy - Psychophysics software in Python. Journal of Neuroscience Methods, 162(1-2), 8-13.
  Peirce, JW (2009) Generating stimuli for neuroscience using PsychoPy. Frontiers in Neuroinformatics, 2:10. doi: 10.3389/neuro.11.010.2008
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
from psychopy import locale_setup, visual, core, data, event, logging, sound, gui
from psychopy.constants import *  # things like STARTED, FINISHED
import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import sin, cos, tan, log, log10, pi, average, sqrt, std, deg2rad, rad2deg, linspace, asarray
from numpy.random import randint, normal, shuffle
import os  # handy system and path functions
import sys # to get file system encoding
from utils import Flicker
from utils import MotionDots
import random
import pandas as pd

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__)).decode(sys.getfilesystemencoding())
os.chdir(_thisDir)

# Store info about the experiment session
expName = u'bio_motion_task'  # from the Builder filename that created this script
expInfo = {'participant':'', 'session':'001', 'Number of Trials':'240'}
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName, order=['participant', 'Number of Trials', 'session'])
if dlg.OK == False: core.quit()  # user pressed cancel
expInfo['expName'] = expName

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s' %(expInfo['participant'], expName)
# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath=None,
    savePickle=True, saveWideText=True,
    dataFileName=filename)
#save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
x=-1
# Start Code - component code to be run before the window creation

# Setup the Window
win = visual.Window(size=(1440, 900), fullscr=True, screen=0, allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, units='height')


# store frame rate of monitor if we can measure it successfully
expInfo['frameRate']=win.getActualFrameRate()
if expInfo['frameRate']!=None:
    frameDur = 1.0/round(expInfo['frameRate'])
else:
    frameDur = 1.0/60.0 # couldn't get a reliable measure so guess

#turn on frame time tracking
#win.setRecordFrameIntervals(True)
#win.saveFrameIntervals(fileName='frames', clear=True)

trigger = Flicker(win, pos=(.75, .45))

# Initialize components for Routine "trial"
trialClock = core.Clock()
ISI = core.StaticPeriod(win=win, screenHz=expInfo['frameRate'], name='ISI')

motion = ['Chop','Crawl','Drive','Peddle','Playpool','Row','Saw','Stir','Sweep','Playtennis','Walk',
        'Cycle','Drink','Jump','Mow','Paint','Pump','Salute','Spade','Wave','Chop_Control',
        'Crawl_Control','Drive_Control','Peddle_Control','Playpool_Control','Row_Control','Saw_Control',
        'Stir_Control','Sweep_Control','Playtennis_Control','Walk_Control','Cycle_Control','Drink_Control',
        'Jump_Control','Mow_Control','Paint_Control','Pump_Control','Salute_Control','Spade_Control','Wave_Control']

motion_color1 = [(0,100,0),(0,100,0),(0,100,0),(0,100,0),(0,100,0),(0,100,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),
                (255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0)]
random.shuffle(motion_color1)

motion_color2 = [(0,100,0),(0,100,0),(0,100,0),(0,100,0),(0,100,0),(0,100,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),
                (255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0),(255,0,0)]
random.shuffle(motion_color2)

motion_color = motion_color1 + motion_color2

random_num = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
            21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
random.shuffle(random_num)

dot_xys = []

Fixation_Cross = visual.TextStim(win=win, ori=0, name='Fixation_Cross',
    text=u'+', font=u'Arial',
    pos=[0, 0], height=0.1, wrapWidth=None,
    color=u'white', colorSpace='rgb', opacity=1,
    depth=-1.0)

#     "How to make sure that you can check whether response is correct based on dots motion"
#     "How to save in additional data about coherence level"
#     "How to save timing data from each of the start times (fixation/cue/start of dots)"

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine

num_trials = int(expInfo['Number of Trials'])
print num_trials

# set up handler to look after randomisation of conditions etc
trials = data.TrialHandler(nReps=num_trials, method='random',
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='trials')
    # if autolog=True, then task doesn't work.

thisExp.addLoop(trials)  # add the loop to the experiment
thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
for thisTrial in trials:
    currentLoop = trials
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial.keys():
            exec(paramName + '= thisTrial.' + paramName)

    #------Prepare to start Routine "trial"-------
    t = 0
    if (x == 39):
        random.shuffle(random_num)
        random.shuffle(motion_color1)
        random.shuffle(motion_color2)
        motion_color = motion_color1 + motion_color2
        x = 0
    else:    
        x += 1
    trialClock.reset()  # clock
    frameN = -1
    routineTimer.add(20.000000)

    # update component parameters for each repeat

    # logic to select:
    # - whether trial type is cued
    # - left or right-moving
    # - coherence of trial
    # Random array with replacement from coherences
    if random_num[x] > 10:
        angle = 1
    if random_num[x] < 11:
        angle = 0
        
    if random_num[x] > 19:
        bio_control = 1
    if random_num[x] < 20:
        bio_control = 0

    file = motion[random_num[x]]
    dot_color=motion_color[random_num[x]]

    dot_stim = MotionDots(win, moviename=file, angle=angle, color = dot_color) 

    fix_cross_duration = 0.8 + 1.2 * np.random.rand()  # uniform random between 1 and 2
    dots_onset = (fix_cross_duration)
    dots_max_duration = 2.

    key_response = event.BuilderKeyResponse()  # create an object of type KeyResponse
    key_response.status = NOT_STARTED
    has_response = False
    begin_ITI = False

    # keep track of which components have finished
    trialComponents = []
    trialComponents.append(ISI)
    trialComponents.append(Fixation_Cross)
    trialComponents.append(dot_stim)
    trialComponents.append(key_response)
    for thisComponent in trialComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED

    #-------Start Routine "trial"-------
    continueRoutine = True
    trial_start_time = globalClock.getTime()
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = trialClock.getTime()
        frameN = frameN + 1.
        # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame

        # *Fixation_Cross* updates
        if t >= 0.0 and Fixation_Cross.status == NOT_STARTED:
            # keep track of start time/frame for later
            Fixation_Cross.tStart = trial_start_time + t   # underestimates by a little under one frame
            Fixation_Cross.frameNStart = frameN  # exact frame index
            Fixation_Cross.setAutoDraw(True)
            offset = trigger.flicker(2)
        if Fixation_Cross.status == STARTED and t >= (0.0 + (fix_cross_duration-win.monitorFramePeriod*.75)): #most of one frame period left
            Fixation_Cross.setAutoDraw(False)

        if t >= fix_cross_duration and dot_stim.status == NOT_STARTED:
            # keep track of start time/frame for later
            dot_stim.tStart = trial_start_time + t  # underestimates by a little under one frame
            dot_stim.frameNStart = frameN  # exact frame index
            dot_stim.setAutoDraw(True)
            offset = trigger.flicker(8)
        if dot_stim.status == STARTED and t >= (dots_max_duration + (fix_cross_duration-win.monitorFramePeriod*.75)): #most of one frame period left
            dot_stim.setAutoDraw(False)
            begin_ITI = True
        # *key_response* updates

        if t >= dots_onset and key_response.status == NOT_STARTED:
            # keep track of start time/frame for later
            key_response.tStart = trial_start_time + t  # underestimates by a little under one frame
            key_response.frameNStart = frameN  # exact frame index
            key_response.status = STARTED
            # keyboard checking is just starting
            win.callOnFlip(key_response.clock.reset)  # t=0 on next screen flip
            event.clearEvents(eventType='keyboard')
        if key_response.status == STARTED and t >= (dots_onset + (dots_max_duration -win.monitorFramePeriod*.75)): #most of one frame period left
            key_response.status = STOPPED
        if key_response.status == STARTED:
            theseKeys = event.getKeys(keyList=['down', 'escape'])
        
        # check for quit:
	    if 'escape' in theseKeys:
	        endExpNow = True
	    if len(theseKeys) > 0:  # at least one key was pressed
	        key_response.keys = theseKeys[-1]  # just the last key pressed
	        key_response.rt = key_response.clock.getTime()
	        has_response = True
	        offset = trigger.flicker(128)

	# Inter-trial interval
        if begin_ITI:
            win.flip()
            ISI.tStart = trial_start_time + t  
            ISI.frameNStart = frameN  # exact frame index
            ISI.start(.5)
            ISI.complete()
            continueRoutine = False  # trial is now over

        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished

        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()

        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()

    #-------Ending Routine "trial"-------
    for thisComponent in trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)

    # store data for trials (TrialHandler)
    trials.addData('file', file)
    trials.addData('color', dot_color)
    trials.addData('angle', angle)
    trials.addData('bio_control', bio_control)
    trials.addData('key_response.keys', key_response.keys)
    trials.addData('key_response.corr', key_response.corr)
    trials.addData('globalClock', globalClock.getTime())
    trials.addData('fix_cross_starttime', Fixation_Cross.tStart)
    trials.addData('dot_stim_starttime', dot_stim.tStart)
    trials.addData('response_starttime', key_response.tStart)

    if key_response.keys != None:  # we had a response
        trials.addData('key_response.rt', key_response.rt)
    if key_response.keys == None:  # we didn't have a response
        trials.addData('key_response.rt', ISI.tStart)
    thisExp.nextEntry()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()
# or data files will save again on exit
win.close()
core.quit()
