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
import random

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__)).decode(sys.getfilesystemencoding())
os.chdir(_thisDir)

# Store info about the experiment session
expName = u'dot_discrimintation_task'  # from the Builder filename that created this script
expInfo = {'participant':'', 'session':'001', 'Number of Trials':'30', 'dots.speed':'.01'}
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName, order=['participant', 'Number of Trials', 'dots.speed', 'session'])
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
dots_speed = float(expInfo['dots.speed'])

# Initialize components for Routine "trial"
trialClock = core.Clock()
ISI = core.StaticPeriod(win=win, screenHz=expInfo['frameRate'], name='ISI')
Fixation_Cross = visual.TextStim(win=win, ori=0, name='Fixation_Cross',
    text=u'+',    font=u'Arial',
    pos=[0, 0], height=0.5, wrapWidth=None,
    color=u'white', colorSpace='rgb', opacity=1,
    depth=-1.0)
cue1 = visual.ImageStim(win=win, name='cue1',units='height', 
    image=u'star.png', mask=None,
    ori=0, pos=[0, 0], size=[0.5, 0.5],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-2.0)
cue2 = visual.ImageStim(win=win, name='cue2',units='height', 
    image=u'heart.png', mask=None,
    ori=0, pos=[0, 0], size=[0.5, 0.5],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-2.0)
cue3 = visual.ImageStim(win=win, name='cue3',units='height', 
    image=u'question.png', mask=None,
    ori=0, pos=[0, 0], size=[0.5, 0.5],
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-2.0)
dots = visual.DotStim(win=win, name='dots',units='height', 
    nDots=100, dotSize=10,
    speed=dots_speed, dir=0.0, coherence=1.0,
    fieldPos=[0.0, 0.0], fieldSize=.7,fieldShape='circle',
    signalDots='same', noiseDots='direction',dotLife=3,
    color=[1.0,1.0,1.0], colorSpace='rgb', opacity=1, depth=-3.0)

FRACTION_AMBIGUOUSLY_CUED = 1/3
coherences = [0, 0.05, 0.1, 0.2, 0.4, 0.8]
is_coherence_hard = [True, True, True, False, False, False]
coherence_options = list(zip(coherences, is_coherence_hard))

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
    trialClock.reset()  # clock 
    frameN = -1
    routineTimer.add(18.000000)
    
    # update component parameters for each repeat

    # logic to select:
    # - whether trial type is cued
    # - left or right-moving
    # - coherence of trial
    # Random array with replacement from coherences
    this_coherence, this_coherence_hard = random.choice(coherence_options)
    dots.coherence = this_coherence

    is_cued = np.random.rand() > FRACTION_AMBIGUOUSLY_CUED  # whether or not trial type is cued
    is_left = np.random.rand() < 0.5  # left vs right
    
    # logic for what type of trial this is:
    if is_cued:  # are we cuing how hard the trial is?
        if not this_coherence_hard:
            this_cue = cue1
        else:
            this_cue = cue2
    else:
        this_cue = cue3  # ambiguous cue
    # Set angle of cue
    if is_left:
        dots.dir = 180
    else:
        dots.dir = 0

    # JMP: add cue timing
    # need to track both onset time and duration for each cue
    #fix_cross_duration = 1 + 1 * np.random.rand()  # uniform random between 1 and 2
    fix_cross_duration = 3.
    cue_onset = 4.
    dots_onset = (fix_cross_duration + cue_onset + 1.)

    key_response = event.BuilderKeyResponse()  # create an object of type KeyResponse
    key_response.status = NOT_STARTED
    # keep track of which components have finished
    trialComponents = []
    trialComponents.append(ISI)
    trialComponents.append(Fixation_Cross)
    trialComponents.append(this_cue)
    trialComponents.append(dots)
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
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Fixation_Cross* updates
        if t >= 0.0 and Fixation_Cross.status == NOT_STARTED:
            # keep track of start time/frame for later
            Fixation_Cross.tStart = trial_start_time + t   # underestimates by a little under one frame
            Fixation_Cross.frameNStart = frameN  # exact frame index
            Fixation_Cross.setAutoDraw(True)
            offset = trigger.flicker_block(2)
        if Fixation_Cross.status == STARTED and t >= (0.0 + (fix_cross_duration-win.monitorFramePeriod*.75)): #most of one frame period left
            Fixation_Cross.setAutoDraw(False)

        if t >= fix_cross_duration and this_cue.status == NOT_STARTED:
            # keep track of start time/frame for later
            this_cue.tStart = trial_start_time + t  # underestimates by a little under one frame
            this_cue.frameNStart = frameN  # exact frame index
            this_cue.setAutoDraw(True)
            offset = trigger.flicker_block(4)
        if this_cue.status == STARTED and t >= (cue_onset+ (fix_cross_duration-win.monitorFramePeriod*.75)): #most of one frame period left
            this_cue.setAutoDraw(False)

        # *dots* updates
        if t >= dots_onset and dots.status == NOT_STARTED:
            # keep track of start time/frame for later
            dots.tStart = trial_start_time + t  # underestimates by a little under one frame
            dots.frameNStart = frameN  # exact frame index
            dots.setAutoDraw(True)
            offset = trigger.flicker_block(8)
        if dots.status == STARTED and t >= (dots_onset +(20-win.monitorFramePeriod*.75)): #most of one frame period left
            dots.setAutoDraw(False)
        
        # *key_response* updates
        if t >= dots_onset and key_response.status == NOT_STARTED:
            # keep track of start time/frame for later
            key_response.tStart = trial_start_time + t  # underestimates by a little under one frame
            key_response.frameNStart = frameN  # exact frame index
            key_response.status = STARTED
            # keyboard checking is just starting
            win.callOnFlip(key_response.clock.reset)  # t=0 on next screen flip
            event.clearEvents(eventType='keyboard')
        if key_response.status == STARTED and t >= (dots_onset + (20-win.monitorFramePeriod*.75)): #most of one frame period left
            key_response.status = STOPPED
        if key_response.status == STARTED:
            theseKeys = event.getKeys(keyList=['left', 'right', 'escape'])

            
            # JMP: this whole section handles keyboard input
            # check for quit:
            if 'escape' in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:  # at least one key was pressed
                key_response.keys = theseKeys[-1]  # just the last key pressed
                key_response.rt = key_response.clock.getTime()
                offset = trigger.flicker_block(16)

                win.flip()
                ISI.tStart = t  # underestimates by a little under one frame
                ISI.frameNStart = frameN  # exact frame index
                ISI.start(1)
                ISI.complete()
                
                # was this 'correct'?
                if dots.dir == 0 and key_response.keys == 'right':
                    key_response.corr = 1
                elif dots.dir == 180 and key_response.keys == 'left':
                    key_response.corr = 1
                else:
                    key_response.corr = 0
                if dots.coherence == 0:
                    key_response.corr = 1
                # a response ends the routine
                continueRoutine = False
        
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
    trials.addData('dots_speed', dots.speed)
    trials.addData('dots.coherence', dots.coherence)
    trials.addData('dots.direction', dots.dir)
    trials.addData('key_response.keys', key_response.keys)
    trials.addData('key_response.corr', key_response.corr)
    trials.addData('globalClock', globalClock.getTime())
    trials.addData('fix_cross_starttime', Fixation_Cross.tStart)
    trials.addData('cue_starttime', this_cue.tStart)
    trials.addData('dots_starttime', dots.tStart)
    trials.addData('response_starttime', key_response.tStart)

    if key_response.keys != None:  # we had a response
        trials.addData('key_response.rt', key_response.rt)
    thisExp.nextEntry()
    
# completed 5 repeats of 'trials'

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort() # or data files will save again on exit
win.close()
core.quit()
