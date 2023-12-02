#!/usr/bin/env python
# encoding: utf-8
"""
Session.py

Created by Tomas HJ Knapen on 2009-11-26.
Copyright (c) 2009 TK. All rights reserved.
"""

from Session import *
from scipy.stats import *
from itertools import *
from Tools.other_scripts.circularTools import *

class RivalryReplaySession(Session):
	def analyzeBehavior(self):
		"""docstring for analyzeBehaviorPerRun"""
		for r in self.scanTypeDict['epi_bold']:
			# do principal analysis, keys vary across dates but taken care of within behavior function
			self.runList[r].behavior()
			# put in the right place
			try:
				ExecCommandLine( 'cp ' + self.runList[r].bO.inputFileName + ' ' + self.runFile(stage = 'processed/behavior', run = self.runList[r], extension = '.dat' ) )
			except ValueError:
				pass
			self.runList[r].behaviorFile = self.runFile(stage = 'processed/behavior', run = self.runList[r], extension = '.dat' )
		
		if 'rivalry' in self.conditionDict:
			self.rivalryBehavior = []
			for r in self.conditionDict['rivalry']:
				self.rivalryBehavior.append([self.runList[r].bO.meanPerceptDuration, self.runList[r].bO.meanTransitionDuration,self.runList[r].bO.meanPerceptsNoTransitionsDuration, self.runList[r].bO.perceptEventsAsArray, self.runList[r].bO.transitionEventsAsArray, self.runList[r].bO.perceptsNoTransitionsAsArray])
				# back up behavior analysis in pickle file
				behAnalysisResults = {'meanPerceptDuration': self.runList[r].bO.meanPerceptDuration, 'meanTransitionDuration': self.runList[r].bO.meanTransitionDuration, 'perceptEventsAsArray': self.runList[r].bO.perceptEventsAsArray, 'transitionEventsAsArray': self.runList[r].bO.transitionEventsAsArray,'perceptsNoTransitionsAsArray':self.runList[r].bO.perceptsNoTransitionsAsArray, 'buttonEvents': self.runList[r].bO.buttonEvents, 'yokedEventsAsArray': np.array(self.runList[r].bO.yokedPeriods), 'halfwayTransitionsAsArray': np.array(self.runList[r].bO.halfwayTransitionsAsArray) }
			
				f = open(self.runFile(stage = 'processed/behavior', run = self.runList[r], postFix = ['behaviorAnalyzer'], extension = '.pickle' ), 'w')
				pickle.dump(behAnalysisResults, f)
				f.close()
			for r in self.conditionDict['replay']:
				self.rivalryBehavior.append([self.runList[r].bO.meanPerceptDuration, self.runList[r].bO.meanTransitionDuration,self.runList[r].bO.meanPerceptsNoTransitionsDuration, self.runList[r].bO.perceptEventsAsArray, self.runList[r].bO.transitionEventsAsArray, self.runList[r].bO.perceptsNoTransitionsAsArray])
				# back up behavior analysis in pickle file
				behAnalysisResults = {'meanPerceptDuration': self.runList[r].bO.meanPerceptDuration, 'meanTransitionDuration': self.runList[r].bO.meanTransitionDuration, 'perceptEventsAsArray': self.runList[r].bO.perceptEventsAsArray, 'transitionEventsAsArray': self.runList[r].bO.transitionEventsAsArray,'perceptsNoTransitionsAsArray':self.runList[r].bO.perceptsNoTransitionsAsArray, 'buttonEvents': self.runList[r].bO.buttonEvents, 'yokedEventsAsArray': np.array(self.runList[r].bO.yokedPeriods), 'halfwayTransitionsAsArray': np.array(self.runList[r].bO.halfwayTransitionsAsArray) }
			
				f = open(self.runFile(stage = 'processed/behavior', run = self.runList[r], postFix = ['behaviorAnalyzer'], extension = '.pickle' ), 'w')
				pickle.dump(behAnalysisResults, f)
				f.close()
			for r in self.conditionDict['replay2']:
				self.rivalryBehavior.append([self.runList[r].bO.meanPerceptDuration, self.runList[r].bO.meanTransitionDuration,self.runList[r].bO.meanPerceptsNoTransitionsDuration, self.runList[r].bO.perceptEventsAsArray, self.runList[r].bO.transitionEventsAsArray, self.runList[r].bO.perceptsNoTransitionsAsArray])
				# back up behavior analysis in pickle file
				behAnalysisResults = {'meanPerceptDuration': self.runList[r].bO.meanPerceptDuration, 'meanTransitionDuration': self.runList[r].bO.meanTransitionDuration, 'perceptEventsAsArray': self.runList[r].bO.perceptEventsAsArray, 'transitionEventsAsArray': self.runList[r].bO.transitionEventsAsArray,'perceptsNoTransitionsAsArray':self.runList[r].bO.perceptsNoTransitionsAsArray, 'buttonEvents': self.runList[r].bO.buttonEvents, 'yokedEventsAsArray': np.array(self.runList[r].bO.yokedPeriods), 'halfwayTransitionsAsArray': np.array(self.runList[r].bO.halfwayTransitionsAsArray) }
				
				f = open(self.runFile(stage = 'processed/behavior', run = self.runList[r], postFix = ['behaviorAnalyzer'], extension = '.pickle' ), 'w')
				pickle.dump(behAnalysisResults, f)
				f.close()
	
	def gatherBehavioralData(self, whichRuns, whichEvents = ['perceptEventsAsArray','transitionEventsAsArray'], sampleInterval = [0,0]):
		data = dict([(we, []) for we in whichEvents])
		timeOffset = 0.0
		for r in whichRuns:
			# behavior for this run, assume behavior analysis has already been run so we can load the results.
			behFile = open(self.runFile(stage = 'processed/behavior', run = self.runList[r], postFix = ['behaviorAnalyzer'], extension = '.pickle'), 'r')
			behData = pickle.load(behFile)
			behFile.close()
			
			niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[r], postFix = ['mcf']))
			TR = niiFile.rtime
			nrTRs = niiFile.timepoints
			
			for we in whichEvents:
				# take data from the time in which we can reliably sample the ERAs
				timeIndices = (behData[we][:,0] > -sampleInterval[0]) * (behData[we][:,0] < -sampleInterval[1] + (TR * nrTRs))
				behData[we] = behData[we][ timeIndices ]
				# implement time offset. 
				behData[we][:,0] = behData[we][:,0] + timeOffset
				data[we].append(behData[we])
				
				
			timeOffset += TR * nrTRs
			
		for we in whichEvents:
			data[we] = np.vstack(data[we])
			
		self.logger.debug('gathered behavioral data from runs %s', str(whichRuns))
		
		return data
	
	def deconvolveEvents(self, roi):
		"""deconvolution analysis on the bold data of rivalry runs in this session for the given roi"""
		self.logger.info('starting deconvolution for roi %s', roi)
		
		roiData = self.gatherRIOData(roi, whichRuns = self.conditionDict['rivalry'] + self.conditionDict['replay'] + self.conditionDict['replay2'], whichMask = 'rivalry_Z' )
		
#		eventData = self.gatherBehavioralData( whichRuns = self.conditionDict['rivalry'] + self.conditionDict['replay'] + self.conditionDict['replay2'], whichEvents = ['perceptEventsAsArray','transitionEventsAsArray','yokedEventsAsArray','halfwayTransitionsAsArray'] )
#		eventArray = [eventData['perceptEventsAsArray'][:,0], eventData['transitionEventsAsArray'][:,0], eventData['yokedEventsAsArray'][:,0], eventData['halfwayTransitionsAsArray'][:,0]]
		eventData = self.gatherBehavioralData( whichRuns = self.conditionDict['rivalry'] + self.conditionDict['replay'] + self.conditionDict['replay2'], whichEvents = ['perceptEventsAsArray','transitionEventsAsArray','yokedEventsAsArray'] )
		eventArray = [eventData['perceptEventsAsArray'][:,0], eventData['transitionEventsAsArray'][:,0], eventData['yokedEventsAsArray'][:,0]]
		
		self.logger.debug('deconvolution analysis with input data shaped: %s', str(roiData.shape))
		# mean data over voxels for this analysis
		decOp = DeconvolutionOperator(inputObject = roiData.mean(axis = 1), eventObject = eventArray)
#		pl.plot(decOp.deconvolvedTimeCoursesPerEventType[0], c = 'r', alpha = 0.75)
#		pl.plot(decOp.deconvolvedTimeCoursesPerEventType[1], c = 'g', alpha = 0.75)
#		pl.plot(decOp.deconvolvedTimeCoursesPerEventType[2], c = 'b', alpha = 0.75)
		pl.plot(decOp.deconvolvedTimeCoursesPerEventType.T)
		
		return decOp.deconvolvedTimeCoursesPerEventType
		
		
	
	def deconvolveEventsFromRois(self, roiArray = ['V1','V2','MT','lingual','superiorparietal','inferiorparietal','insula'], eventType = 'perceptEventsAsArray'):
		res = []
		fig = pl.figure(figsize = (3.5,10))
		pl.subplots_adjust(hspace=0.4)
		for r in range(len(roiArray)):
			s = fig.add_subplot(len(roiArray),1,r+1)
			if r == 0:
				s.set_title(self.subject.initials + ' deconvolution', fontsize=12)
			res.append(self.deconvolveEvents(roiArray[r]))
			s.set_xlabel(roiArray[r], fontsize=9)
		return res
	
	def eventRelatedAverageEvents(self, roi, eventType = 'perceptEventsAsArray', whichRuns = None, whichMask = '_transStateGTdomState', color = 'k', signal = 'mean'):
		"""eventRelatedAverage analysis on the bold data of rivalry runs in this session for the given roi"""
		self.logger.info('starting eventRelatedAverage for roi %s', roi)
		
		res = []
		
		roiData = self.gatherRIOData(roi, whichRuns = whichRuns, whichMask = whichMask )
		eventData = self.gatherBehavioralData( whichRuns = whichRuns, whichEvents = ['perceptEventsAsArray','transitionEventsAsArray','yokedEventsAsArray'], sampleInterval = [-6,21] )
		
		# split out two types of events
#		[ones, twos] = [np.abs(eventData[eventType][:,2]) == 1, np.abs(eventData[eventType][:,2]) == 2]
#		all types of transition/percept events split up, both types and beginning/end separately
#		eventArray = [eventData[eventType][ones,0], eventData[eventType][ones,0] + eventData[eventType][ones,1], eventData[eventType][twos,0], eventData[eventType][twos,0] + eventData[eventType][twos,1]]
		
#		combine across percepts types, but separate onsets/offsets
#		eventArray = [eventData[eventType][:,0], eventData[eventType][:,0] + eventData[eventType][:,1]]
		
#		separate out different percepts - looking at onsets
# 		eventArray = [eventData[eventType][ones,0], eventData[eventType][twos,0]]
		
		# just all the onsets
		# take also the half-way transitions as events
		
		
		eventArray = [eventData[eventType][:,0]]
		
		self.logger.debug('eventRelatedAverage analysis with input data shaped: %s, and %s events of type %s', str(roiData.shape), str(eventData[eventType].shape[0]), eventType)
		
		smoothingWidth = 7
		f = np.array([pow(math.e, -(pow(x,2.0)/(smoothingWidth))) for x in np.linspace(-smoothingWidth,smoothingWidth,smoothingWidth*2.0)])
		
		# mean or std data over voxels for this analysis
		if signal == 'mean':
			roiData = roiData.mean(axis = 1)
		elif signal == 'std':
			roiData = roiData.std(axis = 1)
		elif signal == 'cv':
			roiData = roiData.std(axis = 1)/roiData.mean(axis = 1)
			
		for e in range(len(eventArray)):
			eraOp = EventRelatedAverageOperator(inputObject = np.array([roiData]), eventObject = eventArray[e], interval = [-2.0,14.0])
			d = eraOp.run(binWidth = 4.0, stepSize = 0.25)
			pl.plot(d[:,0], d[:,1], c = color, alpha = 0.75)
#			pl.plot(d[:,0], np.convolve(f/f.sum(), d[:,1], 'same'), c = color, alpha = 0.6)
			res.append(d)
		return res
			
	
	def eventRelatedAverageEventsFromRois(self, roiArray = ['V1','V2','MT','lingual','superiorparietal','inferiorparietal','insula'], whichMask = '_transStateGTdomState', signal = 'mean'):
		evRes = []
		fig = pl.figure(figsize = (3.5,10))
		
		pl.subplots_adjust(hspace=0.4)
		for r in range(len(roiArray)):
			evRes.append([])
			s = fig.add_subplot(len(roiArray),1,r+1)
			if r == 0:
				s.set_title(self.subject.initials + ' averaged ' + signal, fontsize=12)
			evRes[r].append(self.eventRelatedAverageEvents(roiArray[r], eventType = 'perceptEventsAsArray', whichRuns = self.conditionDict['rivalry'] + self.conditionDict['replay'] + self.conditionDict['replay2'], whichMask = whichMask, color = 'r', signal = signal))
			evRes[r].append(self.eventRelatedAverageEvents(roiArray[r], eventType = 'transitionEventsAsArray', whichRuns = self.conditionDict['rivalry'] + self.conditionDict['replay'] + self.conditionDict['replay2'], whichMask = whichMask, color = 'g', signal = signal))
#			evRes[r].append(self.eventRelatedAverageEvents(roiArray[r], eventType = 'yokedEventsAsArray', whichRuns = self.conditionDict['replay'], whichMask = whichMask, color = 'b', signal = signal))
#			evRes[r].append(self.eventRelatedAverageEvents(roiArray[r], eventType = 'yokedEventsAsArray', whichRuns = self.conditionDict['replay2'], whichMask = whichMask, color = 'k', signal = signal))
#			evRes[r].append(self.eventRelatedAverageEvents(roiArray[r], eventType = 'halfwayTransitionsAsArray', whichRuns = self.conditionDict['rivalry'] + self.conditionDict['replay'] + self.conditionDict['replay2'], whichMask = whichMask, color = 'k'))
			s.set_xlabel(roiArray[r], fontsize=9)
			if signal == 'mean':
				pl.xlim([0,12])
			elif signal == 'std':
				s.axis([0,12,0.85,0.985])
		
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'event-related_' + signal + '.pdf'))
		return evRes
	
	def runTransitionGLM(self, perRun = False, acrossRuns = True):
		"""
		Take all transition events and use them as event regressors
		Run FSL on this
		"""
		if perRun:
			for condition in ['rivalry', 'replay', 'replay2']:
				for run in self.conditionDict[condition]:
					# create the event files
					for eventType in ['perceptEventsAsArray','transitionEventsAsArray','yokedEventsAsArray']:
						eventData = self.gatherBehavioralData( whichEvents = [eventType], whichRuns = [run] )
						eventName = eventType.split('EventsAsArray')[0]
						dfFSL = np.ones((eventData[eventType].shape[0],3)) * [1.0,0.1,1]
						dfFSL[:,0] = eventData[eventType][:,0]
						np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = eventName, extension = '.evt'), dfFSL, fmt='%4.2f')
						# also make files for the end of each event.
						dfFSL[:,0] = eventData[eventType][:,0] + eventData[eventType][:,1]
						np.savetxt(self.runFile( stage = 'processed/mri', run = self.runList[run], base = eventName, postFix = ['end'], extension = '.evt'), dfFSL, fmt='%4.2f')
					# remove previous feat directories
					try:
						self.logger.debug('rm -rf ' + self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf'], extension = '.feat'))
						os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf'], extension = '.feat'))
						os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf'], extension = '.fsf'))
					except OSError:
						pass
					# this is where we start up fsl feat analysis after creating the feat .fsf file and the like
					thisFeatFile = '/Users/tk/Documents/research/analysis_tools/Tools/other_scripts/transition.fsf'
					REDict = {
					'---NR_FRAMES---':str(NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf'])).timepoints),
					'---FUNC_FILE---':self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf']), 
					'---ANAT_FILE---':os.path.join(os.environ['SUBJECTS_DIR'], self.subject.standardFSID, 'mri', 'bet', 'T1_bet' ), 
					'---TRANS_ON_FILE---':self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'transition', extension = '.evt'),
					'---TRANS_OFF_FILE---':self.runFile( stage = 'processed/mri', run = self.runList[run], base = 'transition', postFix = ['end'], extension = '.evt')
					}
					featFileName = self.runFile(stage = 'processed/mri', run = self.runList[run], extension = '.fsf')
					featOp = FEATOperator(inputObject = thisFeatFile)
					if run == self.conditionDict['replay2'][-1]:
						featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = True )
					else:
						featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = False )
					# run feat
					featOp.execute()
		# group GLM
		if acrossRuns:
			nrFeats = len(self.scanTypeDict['epi_bold'])
			inputFeats = [self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf'], extension = '.feat') for run in self.scanTypeDict['epi_bold']]
			inputRules = '';	evRules = '';	groupRules = '';
			for i in range(nrFeats):
				inputRules += 'set feat_files(' + str(i+1) + ') "' + inputFeats[i] + '"\n' 
				evRules += 'set fmri(evg' + str(i+1) + '.1) 1\n'
				groupRules += 'set fmri(groupmem.' + str(i+1) + ') 1\n'
				
			try:
				os.mkdir(self.stageFolder(stage = 'processed/mri/feat'))
			except OSError:
				pass
			# this is where we start up fsl feat analysis after creating the feat .fsf file and the like
			thisFeatFile = '/Users/tk/Documents/research/analysis_tools/Tools/other_scripts/acrossRuns.fsf'
			REDict = {
			'---NR_RUNS---':str(i+1),
			'---INPUT_RULES---':inputRules, 
			'---OUTPUT_FOLDER---': os.path.join(self.stageFolder(stage = 'processed/mri/feat'), 'acrossRuns'), 
			'---EV_RULES---':evRules,
			'---GROUP_RULES---':groupRules
			}
			featFileName = os.path.join( self.stageFolder(stage = 'processed/mri/feat'), 'acrossRuns.fsf')
			featOp = FEATOperator(inputObject = thisFeatFile)
			featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = False )
			featOp.execute()
			
	
#			def coherenceAnalysis(self, rois = [['pericalcarine', 'lateraloccipital','lingual'],['inferiorparietal', 'superiorparietal','cuneus','precuneus','supramarginal'],['insula','superiortemporal', 'parsorbitalis','parstriangularis','parsopercularis','rostralmiddlefrontal'],['caudalmiddlefrontal','precentral', 'superiorfrontal']], labels = ['occipital','parietal','inferiorfrontal','fef']):
	def coherenceAnalysis(self, roiArray = [['pericalcarine', 'lateraloccipital','lingual'],['inferiorparietal', 'superiorparietal','cuneus','precuneus'],['supramarginal'],['superiortemporal', 'parsorbitalis','parstriangularis','parsopercularis','caudalmiddlefrontal','precentral'], ['superiorfrontal'], ['rostralmiddlefrontal']], labels = ['occ','par','tpj','inffr','fef','dlpfc'], acrossAreas = False, acrossConditions = True):
		
		#Import the time-series objects: 
		from nitime.timeseries import TimeSeries 
		#Import the analysis objects:
		from nitime.analysis import CoherenceAnalyzer
		#Import utility functions:
		import nitime.viz as viz
		from nitime.viz import drawmatrix_channels,drawgraph_channels
		
		# parameters
		TR=2.0
		f_lb = 0.1
		f_ub = 0.5
		
		self.labels = labels
		roi_names = np.array(self.labels)
		
		self.coherences = []
		self.delays = []
		self.Cs = []
		
		if acrossAreas:
		
			plotNr = 1
			for (ts, runs) in zip([[4,64],[69,129],[69,129]],[self.scanTypeDict['epi_bold'],self.conditionDict['replay'],self.conditionDict['replay2']]):
				roiData = []
				for roi in roiArray:
					thisRoiData = self.gatherRIOData(roi, runs, whichMask = '_rivalry_Z', timeSlices = ts)
					roiData.append(thisRoiData.mean(axis = 1))
				roiData = np.array(roiData)
			#	self.conditionCoherenceDict.append( np.array(roiData) )
				n_samples = roiData.shape[1]
			
				T = TimeSeries(roiData,sampling_interval=TR)
				T.metadata['roi'] = roi_names
			
				C = CoherenceAnalyzer(T, unwrap_phases = True)
				freq_idx = np.where((C.frequencies>f_lb) * (C.frequencies<f_ub))[0]
			
				self.coherences.append(np.mean(C.coherence[:,:,freq_idx],-1))
				self.delays.append(np.mean(C.delay[:,:,freq_idx],-1))
				self.Cs.append(C)
	#			drawmatrix_channels(self.delays[-1],roi_names, title = self.subject.initials + '\n Delay ' + ['rivalry','instantaneous replay','duration-matched replay'][plotNr - 1])
	#			drawmatrix_channels(self.coherences[-1],roi_names, title = self.subject.initials + '\n Coherence ' + ['rivalry','instantaneous replay','duration-matched replay'][plotNr - 1])
				plotNr += 1
		
			f = pl.figure(figsize = (5,7))
			drawmatrix_channels(self.delays[0],labels, size = (5,7), fig = f, title = self.subject.initials + '\n Delay rivalry')
			f = pl.figure(figsize = (5,7))
			drawmatrix_channels(self.delays[1],labels, size = (5,7), fig = f, title = self.subject.initials + '\n Delay instant replay')
			f = pl.figure(figsize = (5,7))
			drawmatrix_channels(self.delays[2],labels, size = (5,7), fig = f, title = self.subject.initials + '\n Delay duration-matched replay')
		
			f = pl.figure(figsize = (5,7))
			drawmatrix_channels(self.delays[0] - self.delays[1],labels, size = (5,7), fig = f, title = self.subject.initials + '\n Delay difference between rivalry and instant replay')
			f = pl.figure(figsize = (5,7))
			drawmatrix_channels(self.delays[0] - self.delays[2],labels, size = (5,7), fig = f, title = self.subject.initials + '\n Delay difference between rivalry and duration-matched replay')
			f = pl.figure(figsize = (5,7))
			drawmatrix_channels(self.delays[2] - self.delays[1],labels, size = (5,7), fig = f, title = self.subject.initials + '\n Delay difference between duration-matched and instant replay')
		
		if acrossConditions:
			self.plotData = []
			fig1 = pl.figure(figsize = (8,3))
			for (counter, runs) in zip([0,1],[self.conditionDict['replay'],self.conditionDict['replay2']]):
				roiData = []
				for roi in roiArray:
					thisRivData = self.gatherRIOData(roi, runs, whichMask = '_rivalry_Z', timeSlices = [4,64]).mean(axis = 1)
					thisRepData = self.gatherRIOData(roi, runs, whichMask = '_rivalry_Z', timeSlices = [69,129]).mean(axis = 1)
				
					roiData.append(thisRivData)
					roiData.append(thisRepData)
				roiData = np.array(roiData)
				n_samples = roiData.shape[1]
			
				T = TimeSeries(roiData,sampling_interval=TR)
				T.metadata['roi'] = np.array([[roi_names[i] + ' Riv', roi_names[i] + ' Rep'] for i in range(len(roi_names))]).ravel()
			
				C = CoherenceAnalyzer(T, unwrap_phases = True)
				freq_idx = np.where((C.frequencies>f_lb) * (C.frequencies<f_ub))[0]
				np.mean(C.coherence[:,:,freq_idx],-1)
				np.mean(C.delay[:,:,freq_idx],-1)
			
		#		f = pl.figure(figsize = (5,7))
		#		drawmatrix_channels(np.mean(C.delay[:,:,freq_idx],-1), T.metadata['roi'], size = (5,7), fig = f, title = self.subject.initials + '\n Rivalry and Replay ' + ['intant', 'dur-match'][counter])
				
				rivrepindices = np.array([np.array([0,1]) + (2 * i) for i in range(6)])
				delay = np.mean(C.delay[:,:,freq_idx],-1)
				plotData = [delay[rivrepindices[i,0],rivrepindices[i,1]] for i in range(len(rivrepindices))]
				
				self.plotData.append(plotData)
	


class RivalryLearningSession(Session):
	def analyzeBehavior(self):
		"""docstring for analyzeBehaviorPerRun"""
		for r in self.scanTypeDict['epi_bold']:
			# do principal analysis, keys vary across dates but taken care of within behavior function
			self.runList[r].behavior()
			# put in the right place
			try:
				ExecCommandLine( 'cp ' + self.runList[r].bO.inputFileName + ' ' + self.runFile(stage = 'processed/behavior', run = self.runList[r], extension = '.pickle' ) )
			except ValueError:
				pass
			self.runList[r].behaviorFile = self.runFile(stage = 'processed/behavior', run = self.runList[r], extension = '.pickle' )
			
		if 'disparity' in self.conditionDict:
			self.disparityPsychophysics = []
			for r in self.conditionDict['disparity']:
				self.disparityPsychophysics.append([self.runList[r].bO.disparities ,self.runList[r].bO.answersPerStimulusValue, self.runList[r].bO.meanAnswersPerStimulusValue, self.runList[r].bO.fit])
				# back up behavior analysis in pickle file
				f = open(self.runFile(stage = 'processed/behavior', run = self.runList[r], postFix = ['behaviorAnalyzer'], extension = '.pickle' ), 'w')
				pickle.dump([self.runList[r].bO.disparities ,self.runList[r].bO.answersPerStimulusValue, self.runList[r].bO.meanAnswersPerStimulusValue,self.runList[r].bO.fit.data], f)
				f.close()
			# repeat fitting across trials
			allFitsData = np.array([d[-1].data for d in self.disparityPsychophysics])
			data = zip(allFitsData[0,:,0],allFitsData[:,:,1].sum(axis = 0),allFitsData[:,:,2].sum(axis = 0))
			pf = BootstrapInference(data, sigmoid = 'gauss', core = 'ab', nafc = 1, cuts = [0.25,0.5,0.75])
			pf.sample()
			GoodnessOfFit(pf)
		
		
		if 'rivalry' in self.conditionDict:
			self.rivalryBehavior = []
			for r in self.conditionDict['rivalry']:
				self.rivalryBehavior.append([self.runList[r].bO.meanPerceptDuration, self.runList[r].bO.meanTransitionDuration,self.runList[r].bO.meanPerceptsNoTransitionsDuration, self.runList[r].bO.perceptEventsAsArray, self.runList[r].bO.transitionEventsAsArray, self.runList[r].bO.perceptsNoTransitionsAsArray])
				# back up behavior analysis in pickle file
				behAnalysisResults = {'meanPerceptDuration': self.runList[r].bO.meanPerceptDuration, 'meanTransitionDuration': self.runList[r].bO.meanTransitionDuration, 'perceptEventsAsArray': self.runList[r].bO.perceptEventsAsArray, 'transitionEventsAsArray': self.runList[r].bO.transitionEventsAsArray,'perceptsNoTransitionsAsArray':self.runList[r].bO.perceptsNoTransitionsAsArray, 'buttonEvents': self.runList[r].bO.buttonEvents, 'yokedEventsAsArray': np.array(self.runList[r].bO.yokedPeriods) }
				
				f = open(self.runFile(stage = 'processed/behavior', run = self.runList[r], postFix = ['behaviorAnalyzer'], extension = '.pickle' ), 'w')
				pickle.dump(behAnalysisResults, f)
				f.close()
			
			firstHalfLength = floor(len(self.conditionDict['rivalry']) / 2.0)
		
			fig = pl.figure()
			s = fig.add_subplot(1,1,1)
			# first series of EPI runs for rivalry learning
	#		with (first) and without (second) taking into account the transitions that were reported.
			pl.scatter(np.arange(firstHalfLength)+0.5, [self.rivalryBehavior[i][0] for i in range(firstHalfLength)], c = 'b', alpha = 0.85)
			pl.scatter(np.arange(firstHalfLength)+0.5, [self.rivalryBehavior[i][2] for i in range(firstHalfLength)], c = 'b', alpha = 0.75, marker = 's')
			
			# all percept events, plotted on top of this
			pl.plot(np.concatenate([(self.rivalryBehavior[rb][5][:,0]/150.0) + rb for rb in range(firstHalfLength)]), np.concatenate([self.rivalryBehavior[rb][5][:,1] for rb in range(firstHalfLength)]), c = 'b', alpha = 0.25)
			# second series of EPI runs
	#		with (first) and without (second) taking into account the transitions that were reported.
			pl.scatter(np.arange(firstHalfLength,len(self.conditionDict['rivalry']))+0.5, [self.rivalryBehavior[i][0] for i in range(firstHalfLength,len(self.conditionDict['rivalry']))], c = 'g', alpha = 0.85)
			pl.scatter(np.arange(firstHalfLength,len(self.conditionDict['rivalry']))+0.5, [self.rivalryBehavior[i][2] for i in range(firstHalfLength,len(self.conditionDict['rivalry']))], c = 'g', alpha = 0.75, marker = 's')
			# all percept events, plotted on top of this
	#		with (first) and without (second) taking into account the transitions that were reported.
			pl.plot(np.concatenate([(self.rivalryBehavior[rb][3][:,0]/150.0) + rb for rb in range(firstHalfLength,len(self.conditionDict['rivalry']))]), np.concatenate([self.rivalryBehavior[rb][3][:,1] for rb in range(firstHalfLength,len(self.conditionDict['rivalry']))]), c = 'g', alpha = 0.25)
			s.axis([-1,13,0,12])
		
			pl.savefig(self.runFile(stage = 'processed/behavior', extension = '.pdf', base = 'duration_summary' ))
		
	
	def gatherBehavioralData(self, whichRuns, whichEvents = ['perceptEventsAsArray','transitionEventsAsArray'], sampleInterval = [0,0]):
		data = dict([(we, []) for we in whichEvents])
		timeOffset = 0.0
		for r in whichRuns:
			# behavior for this run, assume behavior analysis has already been run so we can load the results.
			behFile = open(self.runFile(stage = 'processed/behavior', run = self.runList[r], postFix = ['behaviorAnalyzer'], extension = '.pickle'), 'r')
			behData = pickle.load(behFile)
			behFile.close()
			
			niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[r], postFix = ['mcf']))
			TR = niiFile.rtime
			nrTRs = niiFile.timepoints
			
			for we in whichEvents:
				# take data from the time in which we can reliably sample the ERAs
				behData[we] = behData[we][ (behData[we][:,0] > -sampleInterval[0]) * (behData[we][:,0] < -sampleInterval[1] + (TR * nrTRs)) ]
				# implement time offset. 
				behData[we][:,0] = behData[we][:,0] + timeOffset
				data[we].append(behData[we])
			
			timeOffset += TR * nrTRs
			
		for we in whichEvents:
			data[we] = np.vstack(data[we])
			
		self.logger.debug('gathered behavioral data from runs %s', str(whichRuns))
		
		return data
		
	
	def deconvolveEvents(self, roi, eventType = 'perceptEventsAsArray'):
		"""deconvolution analysis on the bold data of rivalry runs in this session for the given roi"""
		self.logger.info('starting deconvolution for roi %s', roi)
		
		roiData = self.gatherRIOData(roi, whichRuns = self.conditionDict['rivalry'] )
		eventData = self.gatherBehavioralData( whichRuns = self.conditionDict['rivalry'] )
		# split out two types of events
		[ones, twos] = [np.abs(eventData[eventType][:,2]) == 1, np.abs(eventData[eventType][:,2]) == 2]
#		all types of transition/percept events split up, both types and beginning/end separately
#		eventArray = [eventData[eventType][ones,0], eventData[eventType][ones,0] + eventData[eventType][ones,1], eventData[eventType][twos,0], eventData[eventType][twos,0] + eventData[eventType][twos,1]]
		
#		combine across percepts types, but separate onsets/offsets
		eventArray = [eventData[eventType][:,0], eventData[eventType][:,0] + eventData[eventType][:,1]]
		
#		separate out different percepts - looking at onsets
#		eventArray = [eventData[eventType][ones,0], eventData[eventType][twos,0]]
		
		self.logger.debug('deconvolution analysis with input data shaped: %s, and %s events of type %s', str(roiData.shape), str(eventData[eventType].shape[0]), eventType)
		# mean data over voxels for this analysis
		colors = ['k','r','g','b']
		decOp = DeconvolutionOperator(inputObject = roiData.mean(axis = 1), eventObject = eventArray)
		pl.plot(decOp.deconvolvedTimeCoursesPerEventType.T)
	
	def deconvolveEventsFromRois(self, roiArray = ['V1','V2','MT','lingual','superiorparietal','inferiorparietal','insula'], eventType = 'perceptEventsAsArray'):
		
		fig = pl.figure(figsize = (3.5,10))
		
		for r in range(len(roiArray)):
			s = fig.add_subplot(len(roiArray),1,r+1)
			self.deconvolveEvents(roiArray[r], eventType = eventType)
			s.set_xlabel(roiArray[r], fontsize=10)
	
	def eventRelatedAverageEvents(self, roi, eventType = 'perceptEventsAsArray', whichRuns = None, color = 'k'):
		"""eventRelatedAverage analysis on the bold data of rivalry runs in this session for the given roi"""
		self.logger.info('starting eventRelatedAverage for roi %s', roi)
		
		roiData = self.gatherRIOData(roi, whichRuns = whichRuns, whichMask = '_zstat_1' )
		eventData = self.gatherBehavioralData( whichRuns = whichRuns, sampleInterval = [-5,20] )
		
		# split out two types of events
		[ones, twos] = [np.abs(eventData[eventType][:,2]) == 1, np.abs(eventData[eventType][:,2]) == 2]
#		all types of transition/percept events split up, both types and beginning/end separately
#		eventArray = [eventData[eventType][ones,0], eventData[eventType][ones,0] + eventData[eventType][ones,1], eventData[eventType][twos,0], eventData[eventType][twos,0] + eventData[eventType][twos,1]]
		
#		combine across percepts types, but separate onsets/offsets
#		eventArray = [eventData[eventType][:,0], eventData[eventType][:,0] + eventData[eventType][:,1]]
		
#		separate out different percepts - looking at onsets
# 		eventArray = [eventData[eventType][ones,0], eventData[eventType][twos,0]]
		
		# just all the onsets
		eventArray = [eventData[eventType][:,0]]
		
		self.logger.debug('eventRelatedAverage analysis with input data shaped: %s, and %s events of type %s', str(roiData.shape), str(eventData[eventType].shape[0]), eventType)
		# mean data over voxels for this analysis
		roiDataM = roiData.mean(axis = 1)
		roiDataVar = roiData.var(axis = 1)
		for e in range(len(eventArray)):
			eraOp = EventRelatedAverageOperator(inputObject = np.array([roiDataM]), eventObject = eventArray[e], interval = [-3.0,15.0])
			eraOpVar = EventRelatedAverageOperator(inputObject = np.array([roiDataVar]), eventObject = eventArray[e], interval = [-3.0,15.0])
			d = eraOp.run(binWidth = 3.0, stepSize = 0.25)
			dV = eraOpVar.run(binWidth = 3.0, stepSize = 0.25)
			pl.plot(d[:,0], d[:,1], c = color, alpha = 0.75)
			pl.plot(dV[:,0], dV[:,1] - dV[:,1].mean(), c = color, alpha = 0.75, ls = '--')
	
	def eventRelatedAverageEventsFromRois(self, roiArray = ['V1','V2','MT','lingual','superiorparietal','inferiorparietal','insula'], eventType = 'transitionEventsAsArray', learningPartitions = None):
		
		fig = pl.figure(figsize = (3.5,10))
		
		for r in range(len(roiArray)):
			s = fig.add_subplot(len(roiArray),1,r+1)
			self.eventRelatedAverageEvents(roiArray[r], eventType = eventType, whichRuns = self.conditionDict['rivalry'])
			s.set_xlabel(roiArray[r], fontsize=10)
			s.axis([-5,17,-0.1,0.1])
			
		# now, for learning...
		# in learning, there were twelve rivalry runs - there were two sequences of six across which there was 'learning', or at least they were of the same type.
		# we'll make 6 separate ERAs for 
		if learningPartitions:
			colors = [(i/float(len(learningPartitions)), 1.0 - i/float(len(learningPartitions)), 0.0) for i in range(len(learningPartitions))]
			
			fig = pl.figure(figsize = (3.5,10))
			pl.subplots_adjust(hspace=0.4)
			for r in range(len(roiArray)):
				s = fig.add_subplot(len(roiArray),1,r+1)
				for (ind, lp) in zip(range(len(learningPartitions)), learningPartitions):
					self.eventRelatedAverageEvents(roiArray[r], eventType = eventType, whichRuns = [self.conditionDict['rivalry'][i] for i in lp], color = colors[ind])
				s.set_xlabel(roiArray[r], fontsize=9)
				s.axis([-5,17,-0.075,0.075])
	
	def prepareTransitionGLM(self, functionals = False):
		"""
		Take all transition events and use them as event regressors
		Make one big nii file that contains all the motion corrected and zscored rivalry data
		Run FSL on this
		"""
		
		if functionals:
			# make nii file
			niiFiles = [NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[r], postFix = ['mcf','hp','Z'])) for r in self.conditionDict['rivalry'] ]
			allNiiFile = NiftiImage(np.concatenate([nf.data for nf in niiFiles]), header = niiFiles[0].header)
			allNiiFile.save( os.path.join( self.conditionFolder( stage = 'processed/mri', run = self.runList[self.conditionDict['rivalry'][0]]), 'all_rivalry.nii.gz') )
		
		eventData = self.gatherBehavioralData( whichRuns = self.conditionDict['rivalry'] )
		
		perceptDataForFSL = np.ones((eventData['perceptEventsAsArray'].shape[0],3)) * [1.0,0.1,1]
		perceptDataForFSL[:,0] = eventData['perceptEventsAsArray'][:,0]
		np.savetxt(os.path.join( self.conditionFolder( stage = 'processed/mri', run = self.runList[self.conditionDict['rivalry'][0]]), 'all_rivalry_percept' + '.evt'), perceptDataForFSL, fmt='%4.2f')
		
		transDataForFSL = np.ones((eventData['transitionEventsAsArray'].shape[0],3)) * [1.0,0.1,1]
		transDataForFSL[:,0] = eventData['transitionEventsAsArray'][:,0]
		np.savetxt(os.path.join( self.conditionFolder( stage = 'processed/mri', run = self.runList[self.conditionDict['rivalry'][0]]), 'all_rivalry_trans' + '.evt'), transDataForFSL, fmt='%4.2f' )
		
		
	
	def eventRelatedDecodingFromRoi(self, roi, eventType = 'perceptEventsAsArray', whichRuns = None, color = 'k'):
		self.logger.info('starting eventRelatedDecoding for roi %s', roi)
		
		roiData = self.gatherRIOData(roi, whichRuns = [self.conditionDict['rivalry'][i] for i in whichRuns] )
		eventData = self.gatherBehavioralData( whichRuns = [self.conditionDict['rivalry'][i] for i in whichRuns] )
		
		from ..Operators.ImageOperator import Design
		
		d = Design(roiData.shape[0], 2.0, subSamplingRatio = 100)
		forTransitionregressor = np.ones((eventData['transitionEventsAsArray'].shape[0],3))
		forTransitionregressor[:,0] = eventData['transitionEventsAsArray'][:,0]
		forTransitionregressor[:,1] = 0.5
		d.addRegressor(forTransitionregressor)
		# percept regressor
		d.addRegressor(np.hstack( (eventData['perceptEventsAsArray'][:,[0,1]], (eventData['perceptEventsAsArray'][:,[2]]-1.5) * 2.0) ))
		d.convolveWithHRF(hrfType = 'singleGamma', hrfParameters = {'a': 6, 'b': 0.9}) # a = 6, b = 0.9
		
		if True:
			from ..Operators.ArrayOperator import DecodingOperator
		
			# use median thresholding for transition feature indexing
			over_median = d.designMatrix[:,0] > np.median(d.designMatrix[:,0])
			under_median = -over_median
		
			om_indices = np.arange(over_median.shape[0])[over_median]
			um_indices = np.arange(over_median.shape[0])[under_median]
		
			nr_runs = np.min([om_indices.shape[0], um_indices.shape[0]])
			run_width = 10
			dec = DecodingOperator(roiData, decoder = 'libSVM', fullOutput = True)
			
			for i in range(nr_runs-run_width):
				testThisRun = (np.arange(nr_runs) >= i) * (np.arange(nr_runs) < i+run_width)
				trainingThisRun = -testThisRun
				trainingDataIndices = np.concatenate(( om_indices[trainingThisRun], um_indices[trainingThisRun] ))
				testDataIndices = np.concatenate(( om_indices[testThisRun], um_indices[testThisRun] ))
				trainingsLabels = np.concatenate(( -np.ones((nr_runs-run_width)), np.ones((nr_runs-run_width)) ))
				testLabels = np.concatenate(( -np.ones((run_width)), np.ones((run_width)) ))
				
				print dec.decode(trainingDataIndices, trainingsLabels, testDataIndices, testLabels)
	
	def convertRetinoMask(self):
		statfile = os.path.join(self.stageFolder(stage = 'processed/mri/disparity/retino.gfeat/cope1.feat/stats/') , 'zstat1.nii.gz')
		print self.conditionDict
		first_mapping_epi_run = os.path.join(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['disparity'][0]], postFix = ['mcf'], extension = '.feat'), 'example_func.nii.gz')
		transformFile = os.path.join(self.stageFolder(stage = 'processed/mri/disparity/') , 'example_func2standard_INV.mat')
		f = FlirtOperator(statfile, referenceFileName = first_mapping_epi_run )
		f.configureApply(transformFile, outputFileName = os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/'), 'z_stat.nii.gz'))
		f.execute()
	

class SphereSession(Session):
	def analyzeBehavior(self):
		"""docstring for analyzeBehaviorPerRun"""
		behOps = []
		for r in self.scanTypeDict['epi_bold']:
			# do principal analysis, keys vary across dates but taken care of within behavior function
			thisFileName = self.runFile(stage = 'processed/behavior/', run = self.runList[r], extension = '.dat')
			bO = SphereBehaviorOperator(thisFileName)
			behOps.append(bO)
		self.behOps = behOps
	
	def registerfeats(self, run_type = 'sphere_presto', postFix = ['mcf']):
		"""run featregapply for all feat direcories in this session."""
		for r in [self.runList[i] for i in self.conditionDict[run_type]]:
			this_feat = self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat')
			self.setupRegistrationForFeat(this_feat)
	
	def mask_stats_to_hdf(self, run_type = 'sphere_presto', postFix = ['mcf']):
		"""
		Create an hdf5 file to populate with the stats and parameter estimates of the feat results
		"""
		
		anatRoiFileNames = subprocess.Popen('ls ' + self.stageFolder( stage = 'processed/mri/masks/anat/' ) + '*' + standardMRIExtension, shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
		self.logger.info('Taking masks ' + str(anatRoiFileNames))
		rois, roinames = [], []
		for roi in anatRoiFileNames:
			rois.append(NiftiImage(roi))
			roinames.append(os.path.split(roi)[1][:-7])
			
		self.hdf5_filename = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[run_type][0]]), run_type + '.hdf5')
		if os.path.isfile(self.hdf5_filename):
			os.system('rm ' + self.hdf5_filename)
		
		if not os.path.isfile(self.hdf5_filename):
			self.logger.info('starting table file ' + self.hdf5_filename)
			h5file = open_file(self.hdf5_filename, mode = "w", title = run_type + " file")
		else:
			self.logger.info('opening table file ' + self.hdf5_filename)
			h5file = open_file(self.hdf5_filename, mode = "a", title = run_type + " file")
			
		for  r in [self.runList[i] for i in self.conditionDict[run_type]]:
			"""loop over runs, and try to open a group for this run's data"""
			this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = r, postFix = postFix))[1]
			this_feat = self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension = '.feat')
			try:
				thisRunGroup = h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
				self.logger.info('data files from ' + this_feat + ' already in ' + self.hdf5_filename)
			except NoSuchNodeError:
				# import actual data
				self.logger.info('Adding group ' + this_run_group_name + ' to this file')
				thisRunGroup = h5file.createGroup("/", this_run_group_name, 'Run ' + str(r.ID) +' imported from ' + this_feat)
				
			"""
			Now, take different stat masks based on the run_type
			"""
			stat_files = {
							'stim_on_T': os.path.join(this_feat, 'stats', 'tstat1.nii.gz'),
							'stim_on_Z': os.path.join(this_feat, 'stats', 'zstat1.nii.gz'),
							'stim_on_cope': os.path.join(this_feat, 'stats', 'cope1.nii.gz'),
							
							'alternation_T': os.path.join(this_feat, 'stats', 'tstat2.nii.gz'),
							'alternation_Z': os.path.join(this_feat, 'stats', 'zstat2.nii.gz'),
							'alternation_cope': os.path.join(this_feat, 'stats', 'cope2.nii.gz'),
							
							'either_F': os.path.join(this_feat, 'stats', 'zfstat1.nii.gz'),
							
							}
							
			# general info we want in all hdf files
			stat_files.update({
								'residuals': os.path.join(this_feat, 'stats', 'res4d.nii.gz'),
								'psc_hpf_data': self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf', 'psc', 'hpf']), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
								'hpf_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'), # 'input_data': os.path.join(this_feat, 'filtered_func_data.nii.gz'),
								# for these final one, we need to pre-setup the retinotopic mapping data
								'polar_phase': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'polar.nii.gz'),
								# and the averaged stats for on/off and alternations
								'stim_on_T_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'on_off_t.nii.gz'),
								'stim_on_Z_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'on_off_z.nii.gz'),
								'stim_on_cope_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'on_off_cope.nii.gz'),
								'alternation_T_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'trans_t.nii.gz'),
								'alternation_Z_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'trans_z.nii.gz'),
								'alternation_cope_gfeat': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'trans_cope.nii.gz'),
								
			})
			# we need the following precaution for the fifth subject with no eccen mapping data
			if os.path.isfile(os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'eccen.nii.gz')):
				stat_files.update({'eccen_phase': os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat'), 'eccen.nii.gz')})
			
			stat_nii_files = [NiftiImage(stat_files[sf]) for sf in stat_files.keys()]
			
			for (roi, roi_name) in zip(rois, roinames):
				try:
					thisRunGroup = h5file.get_node(where = "/" + this_run_group_name, name = roi_name, classname='Group')
				except NoSuchNodeError:
					# import actual data
					self.logger.info('Adding group ' + this_run_group_name + '_' + roi_name + ' to this file')
					thisRunGroup = h5file.createGroup("/" + this_run_group_name, roi_name, 'Run ' + str(r.ID) +' imported from ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix))
					
				for (i, sf) in enumerate(stat_files.keys()):
					# loop over stat_files and rois
					# to mask the stat_files with the rois:
					imO = ImageMaskingOperator( inputObject = stat_nii_files[i], maskObject = roi, thresholds = [0.0] )
					these_roi_data = imO.applySingleMask(whichMask = 0, maskThreshold = 0.0, nrVoxels = False, maskFunction = '__gt__', flat = True)
					h5file.createArray(thisRunGroup, sf.replace('>', '_'), these_roi_data.astype(np.float32), roi_name + ' data from ' + stat_files[sf])
		h5file.close()
	
	def hdf5_file(self, run_type = 'sphere_presto'):
		self.hdf5_filename = os.path.join(self.conditionFolder(stage = 'processed/mri', run = self.runList[self.conditionDict[run_type][0]]), run_type + '.hdf5')
		if not os.path.isfile(self.hdf5_filename):
			self.logger.info('no table file ' + self.hdf5_filename + 'found for stat mask')
			return None
		else:
			# self.logger.info('opening table file ' + self.hdf5_filename)
			h5file = open_file(self.hdf5_filename, mode = "r", title = run_type + " file")
		return h5file
	
	def roi_data_from_hdf(self, h5file, run, roi_wildcard, data_type, postFix = ['mcf']):
		"""
		drags data from an already opened hdf file into a numpy array, concatenating the data_type data across voxels in the different rois that correspond to the roi_wildcard
		"""
		this_run_group_name = os.path.split(self.runFile(stage = 'processed/mri', run = run, postFix = postFix))[1]
		try:
			thisRunGroup = h5file.get_node(where = '/', name = this_run_group_name, classname='Group')
			# self.logger.info('group ' + self.runFile(stage = 'processed/mri', run = run, postFix = postFix) + ' opened')
			roi_names = []
			for roi_name in h5file.iterNodes(where = '/' + this_run_group_name, classname = 'Group'):
				if len(roi_name._v_name.split('.')) > 1:
					hemi, area = roi_name._v_name.split('.')
					if roi_wildcard == area:
						roi_names.append(roi_name._v_name)
			if len(roi_names) == 0:
				self.logger.info('No rois corresponding to ' + roi_wildcard + ' in group ' + this_run_group_name)
				return None
		except NoSuchNodeError:
			# import actual data
			self.logger.info('No group ' + this_run_group_name + ' in this file')
			return None

		all_roi_data = []
		for roi_name in roi_names:
			thisRoi = h5file.get_node(where = '/' + this_run_group_name, name = roi_name, classname='Group')
			all_roi_data.append( eval('thisRoi.' + data_type + '.read()') )
		all_roi_data_np = np.hstack(all_roi_data).T
		return all_roi_data_np
	
	def deconvolve_roi(self, roi, thresholds = [[2.3, 30.3],[-30.3, -2.3]], mask_type = 'stim_on_Z', s = None):
		"""
		run deconvolution analysis on the input (mcf_psc_hpf) data that is stored in the reward hdf5 file. 
		Event data will be extracted from the .txt fsl event files used for the initial glm.
		roi argument specifies the region from which to take the data.
		"""
		self.gatherBehavioralData()
		# check out the duration of these runs, assuming they're all the same length.
		niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['sphere_presto'][0]]))
		tr, nr_trs = niiFile.rtime, niiFile.timepoints
		run_duration = tr * nr_trs
		
		h5file = self.hdf5_file('sphere_presto')
		cond_labels = ['inside','outside']
		roi_data = []
		nr_runs = 0
		for r in [self.runList[i] for i in self.conditionDict['sphere_presto']]:
			roi_data.append(self.roi_data_from_hdf(h5file, r, roi, 'psc_hpf_data'))
			nr_runs += 1
			
		demeaned_roi_data = []
		for rd in roi_data:
			demeaned_roi_data.append( (rd.T - rd.mean(axis = 1)).T )
		roi_data_per_run = demeaned_roi_data
		roi_data = np.hstack(demeaned_roi_data)
		# print roi_data.shape
		
		# mask data
		mask_data = np.array([self.roi_data_from_hdf(h5file, self.runList[i], roi, mask_type) for i in self.conditionDict['sphere_presto']]).mean(axis = 0)
		if mask_type == 'eccen_phase':
			# mask_data = np.fmod(mask_data[:,0] + 2 * pi, 2 * pi)
			cond_labels = [str(t) for t in thresholds.T[0]]
		# thresholding of mapping data stat values
		masks = np.array([((mask_data > thr[0]) * (mask_data < thr[1])).ravel() for thr in thresholds])
		# print masks.shape, [m.sum() for m in masks]
		
		timeseries = np.array([roi_data[m].mean(axis = 0) for m in masks])
		# print timeseries.shape
		
		if s == None:
			fig = pl.figure(figsize = (9, 3.5))
			s = fig.add_subplot(111)
		s.axhline(0, -10, 30, linewidth = 0.25)
		
		colors = [np.array([1.0,0.0,0.0]) * graylevel for graylevel in np.linspace(0.0, 1.0, len(thresholds))]
		
		# event_data = np.array([self.allTransitions[self.allTransitions[:,0] < self.allTransitions[:,0].mean(),1], self.allTransitions[self.allTransitions[:,0] > self.allTransitions[:,0].mean(),1]])
		# event_data = [self.allTransitions[:,1], self.allStimOnsets[:,1]]
		event_data = self.allTransitions[:,1]
		interval = [0.0,16.0]
				
		decos = [DeconvolutionOperator(inputObject = t, eventObject = [event_data], TR = tr, deconvolutionSampleDuration = tr, deconvolutionInterval = interval[1]) for t in timeseries]
		erops = [EventRelatedAverageOperator(inputObject = np.array([t]), TR = 0.65, eventObject = event_data, interval = [-3.0,interval[1]]) for t in timeseries]
		all_res = []
		for i, d in enumerate(decos):
			# pl.plot(np.linspace(interval[0],interval[1],d.deconvolvedTimeCoursesPerEventType.shape[1]), d.deconvolvedTimeCoursesPerEventType[0], c = colors[i], alpha = 1.0, label = cond_labels[i])
			# pl.plot(np.linspace(interval[0],interval[1],d.deconvolvedTimeCoursesPerEventType.shape[1]), d.deconvolvedTimeCoursesPerEventType[1], c = colors[i], alpha = 1.0, linestyle = '--')
			zero_index = np.arange(erops[i].intervalRange.shape[0])[np.abs(erops[i].intervalRange).min() == np.abs(erops[i].intervalRange)]
			res = erops[i].run(binWidth = 4.55, stepSize = 0.325)
			all_res.append(res)
			pl.plot(res[:,0], res[:,1]-res[zero_index,1], c = colors[i], alpha = 0.75, label = cond_labels[i])
			s.fill_between(res[:,0], res[:,1]-res[zero_index,1] + (res[:,2] / np.sqrt(res[:,3])), res[:,1]-res[zero_index,1] - (res[:,2] / np.sqrt(res[:,3])), color = colors[i], alpha = 0.1)
			# print d.ratio, d.rawDeconvolvedTimeCourse, d.designMatrix.shape
		# deco_per_run = []
		# for i, rd in enumerate(roi_data_per_run):
		# 	event_data_this_run = event_data_per_run[i] - i * run_duration
		# 	deco = DeconvolutionOperator(inputObject = rd[mapping_mask,:].mean(axis = 0), eventObject = event_data[:], TR = tr, deconvolutionSampleDuration = tr/2.0, deconvolutionInterval = interval[1])
		# 	deco_per_run.append(deco.deconvolvedTimeCoursesPerEventType)
		# deco_per_run = np.array(deco_per_run)
		# mean_deco = deco_per_run.mean(axis = 0)
		# std_deco = 1.96 * deco_per_run.std(axis = 0) / sqrt(len(roi_data_per_run))
		# for i in range(0, mean_deco.shape[0]):
		# 	# pl.plot(np.linspace(interval[0],interval[1],mean_deco.shape[1]), mean_deco[i], ['b','b','g','g'][i], alpha = [0.5, 1.0, 0.5, 1.0][i], label = cond_labels[i])
		# 	s.fill_between(np.linspace(interval[0],interval[1],mean_deco.shape[1]), time_signals[i] + std_deco[i], time_signals[i] - std_deco[i], color = ['b','b','g','g'][i], alpha = 0.3 * [0.5, 1.0, 0.5, 1.0][i])
		# 	
		s.set_title(self.subject.initials + '_' + roi + ' ' + mask_type)
		s.set_xlabel('time [s]')
		s.set_ylabel('% signal change')
		s.set_xlim([interval[0]-1.5, interval[1]-1.5])
		leg = s.legend(fancybox = True)
		leg.get_frame().set_alpha(0.5)
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(3.5)  # the legend line width
				
		h5file.close()
		
		pl.draw()
		if s != None:
			pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), roi + '_' + mask_type + '.pdf'))
		
		return all_res
	
	def deconvolve(self, threshold = 3.0, rois = ['V1', 'V2', 'V3', 'V3AB', 'V4', 'LO12'], analysis_type = 'deconvolution'):
		fig = pl.figure(figsize = (9, 1 + len(rois) * 3.5))
		all_res = []
		for roi in rois:
			print roi
			bin_starts = np.linspace(-pi, pi/2, 5) + 0.001
			# self.deconvolve_roi(roi, np.array([bin_starts, pi/2 + bin_starts]).T, mask_type = 'eccen_phase')
			# self.deconvolve_roi(roi, thresholds =[[2.3, 30.3]], mask_type = 'stim_on_Z_gfeat')
			s = fig.add_subplot(len(rois),1,rois.index(roi)+1)
			all_res.append(self.deconvolve_roi(roi, thresholds = [[3.3, 30.3],[-30.3, -1.5]], mask_type = 'stim_on_Z_gfeat', s = s))
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'stim_on_Z_gfeat.pdf'))
		np.save(os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), 'stim_on_Z_gfeat.npy'), np.array(all_res))
	
	def gatherBehavioralData(self, sampleInterval = [0,0]):
		
		trans = []
		percepts = []
		stimOnsets = []
		timeOffset = 0.0
		for (r, i) in zip(self.scanTypeDict['epi_bold'], range(len(self.scanTypeDict['epi_bold']))):
			# behavior for this run, assume behavior analysis has already been run so we can load the results.
			niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[r], postFix = ['mcf']))
			timeOffset = i * niiFile.rtime * niiFile.timepoints
			
			self.rtime = round(niiFile.rtime * 100.0) / 100.0 # rtime precision is 10 ms
			self.timepointsPerRun = niiFile.timepoints
			self.runDuration = self.rtime * self.timepointsPerRun
			
			transitionsThisRun = self.behOps[i].buttonEvents[1:-1].copy() # throw away last transitions and first transitions because they correlate with stimulus onset.
			transitionsThisRun[:,1] = transitionsThisRun[:,1] + timeOffset
			
			onsetsThisRun = [1, timeOffset + 16.0, timeOffset + 16.5]
			
			perceptsThisRun = self.behOps[i].percepts	# keep all percepts
			perceptsThisRun[:,[1,2]] = perceptsThisRun[:,[1,2]] + timeOffset
			
			trans.append(transitionsThisRun)
			percepts.append(perceptsThisRun)
			stimOnsets.append(onsetsThisRun)
		self.allTransitions = np.vstack(trans)
		self.allPercepts = np.vstack(percepts)
		self.allStimOnsets = np.vstack(stimOnsets)
		
		self.logger.debug('gathered behavioral data from all runs')
		
		return self.allTransitions, self.allPercepts
	
	def setupFeatEventFiles(self, buffer_time = 16):
		"""creates event text files for transition events and stim on period"""
		for (r, i) in zip(self.scanTypeDict['epi_bold'], range(len(self.scanTypeDict['epi_bold']))):
			thisNiiFile = self.runFile(stage = 'processed/mri', run = self.runList[r], postFix = ['mcf'])
			thisEvtFile = self.runFile(stage = 'processed/mri', run = self.runList[r], postFix = ['mcf', 'evt'], extension = '.txt')
			thisOoFile = self.runFile(stage = 'processed/mri', run = self.runList[r], postFix = ['mcf', 'onoff'], extension = '.txt')
			
			niiFile = NiftiImage(thisNiiFile)
			runDuration = niiFile.rtime * niiFile.timepoints
			
			onsets = self.behOps[i].buttonEvents[1:-1,1]
			fsfEvtData = np.array([[o, 0.5, 1] for o in onsets])
			fsfOoData = np.array([[buffer_time, runDuration-buffer_time, 1]])
			
			np.savetxt(thisEvtFile, fsfEvtData, fmt = '%3.2f', delimiter = '\t')
			np.savetxt(thisOoFile, fsfOoData, fmt = '%3.2f', delimiter = '\t')
	
	def runTransitionFeats(self):
		# remove previous feat directories
		for (r, i) in zip(self.scanTypeDict['epi_bold'], range(len(self.scanTypeDict['epi_bold']))):
			try:
				self.logger.debug('rm -rf ' + self.runFile(stage = 'processed/mri', run = self.runList[r], postFix = ['mcf'], extension = '.feat'))
				os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = self.runList[r], postFix = ['mcf'], extension = '.feat'))
				os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = self.runList[r], postFix = ['mcf'], extension = '.fsf'))
			except OSError:
				pass
				
			niiFile = NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[r], postFix = ['mcf']))
			runDuration = niiFile.rtime * niiFile.timepoints
			# this is where we start up fsl feat analysis after creating the feat .fsf file and the like
			# thisFeatFile = '/Volumes/7.2_DD/7T/analysis/trans_onoff_one_run.fsf'
			thisFeatFile = '/Volumes/HDD/research/projects/rivalry_fMRI/7T/analysis/one_run.fsf'
			REDict = {
			'---TR---': 			str(niiFile.rtime),
			'---NR_FRAMES---': 		str(niiFile.timepoints),
			'---NII_FILE---': 		self.runFile(stage = 'processed/mri', run = self.runList[r], postFix = ['mcf']), 
			# '---ONOFF_FILE---': 	self.runFile(stage = 'processed/mri', run = self.runList[r], postFix = ['mcf', 'onoff'], extension = '.txt'), 
			'---TRANS_FILE---': 	self.runFile(stage = 'processed/mri', run = self.runList[r], postFix = ['mcf', 'evt'], extension = '.txt')
			}
			featFileName = self.runFile(stage = 'processed/mri', run = self.runList[r], extension = '.fsf')
			featOp = FEATOperator(inputObject = thisFeatFile)
			if i == range(len(self.scanTypeDict['epi_bold']))[-1]:
				featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = True )
			else:
				featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = False )
			self.logger.debug('Running feat from ' + thisFeatFile + ' as ' + featFileName)
			# run feat
			featOp.execute()
	
	def correlate_data_from_run(self, run, rois = ['V1', 'V2', 'V3', 'V4', 'V3A'], data_pairs = [['on_off_cope', 'trans_cope'], ['eccen_phase','trans_cope'], ['eccen_phase','on_off_cope']], plot = True, postFix = ''):
		"""
		correlates two types of data from regions of interest with one another, but more generally than the other function. 
		This function allows you to specify from what file and what type of stat you are going to correlate with one another.
		Specifically, the data_pairs argument is a list of two-item lists which specify the to be correlated stats
		"""
		from scipy import stats
		h5file = self.hdf5_file()
		corrs = np.zeros((len(rois), len(data_pairs)))
		colors = ['r', 'g', 'b', 'y', 'm', 'c']
		if h5file != None:
			# there was a file and it has data in it
			if plot:	
				fig = pl.figure(figsize = (len(rois)*2, len(data_pairs) * 3))
				nr_plots = 1
			for roi in rois:
				for i in range(len(data_pairs)):
					cope1 = self.roi_data_from_hdf(h5file, run, roi, data_pairs[i][0])
					cope2 = self.roi_data_from_hdf(h5file, run, roi, data_pairs[i][1])
					if cope1 != None and cope2 != None:
						if plot:
							s = fig.add_subplot(len(rois), len(data_pairs), nr_plots)
							nr_plots += 1
							s.set_title(roi, fontsize=9)
							nonzeros = (np.abs(cope1[:,0]) > 0.1) * (np.abs(cope2[:,0]) > 0.1)
							# pull phases straight, or not
							# if data_pairs[i][0].split('_')[-1] == 'phase':
							# 	cope1[:,0] = np.fmod(cope1[:,0] + 2 * pi, 2 * pi)
							# if data_pairs[i][1].split('_')[-1] == 'phase':
							# 	cope1[:,0] = np.fmod(cope2[:,0] + 2 * pi, 2 * pi)
								
							# fit linear trend
							(ar,br)=np.polyfit(cope1[nonzeros,0], cope2[nonzeros,0], 1)
							xr=np.polyval([ar,br],cope1[nonzeros,0])
							
							# smoothing the data
							order = np.argsort(cope1[nonzeros,0])
							smooth_width = round(nonzeros.sum() / 10)
							kern = stats.norm.pdf( np.linspace(-3.25,3.25,smooth_width) )
							kern = kern / kern.sum()
							cope1_s = np.convolve( cope1[nonzeros,0][order], kern, 'valid' )
							cope2_s = np.convolve( cope2[nonzeros,0][order] , kern, 'valid' )
							
							# plotting the data
							pl.plot(cope1[nonzeros,0], cope2[nonzeros,0], marker = 'o', ms = 6, mec = 'w', c = colors[i], mew = 1.5, alpha = 0.0625, linewidth = 0) # , alpha = 0.25
							pl.plot(cope1[nonzeros,0], xr, colors[i] + '-', alpha = 0.5, linewidth = 3.5)
							pl.plot(cope1_s, cope2_s, colors[i] + '--', alpha = 0.5, linewidth = 3.5)
							if rois.index(roi) == len(rois)-1:
								s.set_xlabel(data_pairs[i][0], fontsize=9)
							# if data_pairs.index(data_pairs[i]) == 0:
							s.set_ylabel(data_pairs[i][1], fontsize=9)
						
						# look at correlations between voxel patterns for the given data
						srs = stats.spearmanr(cope1, cope2)
						corrs[rois.index(roi), i] = srs[0]
					else:
						self.logger.info('No data to correlate for ' + str(data_pairs[i]) + ' ' + str(roi))
		if plot:
			pl.draw()
			pdf_file_name = os.path.join(self.stageFolder(stage = 'processed/mri/figs/'), self.subject.initials + '_scatter_' + str(run.ID) + '_' + postFix + '.pdf')
			pl.savefig(pdf_file_name)
		h5file.close()
		return corrs
	
	def correlate_data(self, rois = ['V1', 'V2', 'V3', 'V4', 'V3A'], data_pairs = [['stim_on_Z', 'alternation_Z'], ['eccen_phase','alternation_Z'], ['eccen_phase','stim_on_Z']], scatter_plots = True):
		"""
		correlate reward run cope values with one another from all reward runs separately.
		"""
		all_corrs = []
		for r in [self.runList[i] for i in self.conditionDict['sphere_presto']]:
			all_corrs.append(self.correlate_data_from_run(run = r, rois = rois, data_pairs = data_pairs, plot = scatter_plots))
		self.correlate_data_from_run(run = r, rois = rois, data_pairs = [['stim_on_Z_gfeat', 'alternation_Z_gfeat'], ['eccen_phase','alternation_Z_gfeat'], ['eccen_phase','stim_on_Z_gfeat']], plot = scatter_plots, postFix = 'gfeat')
		cs = np.array(all_corrs)
	
	def take_retinotopic_data_from_run(self, run, rois = ['V1', 'V2', 'V3', 'V4', 'V3A'], values = ['alternation_Z'], nr_bins = {'eccen': 3, 'polar': 4}, offsets = {'eccen': 0.0, 'polar': 0.0}):
		h5file = self.hdf5_file()
		
		data = []
		if h5file != None:
			# there was a file and it has data in it
			for roi in rois:
				data.append([])
				# take data
				eccen_data = self.roi_data_from_hdf(h5file, run, roi, 'eccen_phase')
				polar_data = self.roi_data_from_hdf(h5file, run, roi, 'polar_phase')
				# take out zeros
				nonzeros = (eccen_data[:,0] != 0.0) + (polar_data[:,0] != 0.0)
				# correct for offsets and rotate the phase values
				eccen_data = positivePhases(eccen_data + offsets['eccen'])
				polar_data = positivePhases(polar_data + offsets['polar'])
				# the edges of the polar and eccen bins
				bin_edges_eccen = np.array([np.linspace(0, 2*pi, nr_bins['eccen'] + 1, endpoint = False)[:-1], np.linspace(0, 2*pi, nr_bins['eccen'] + 1, endpoint = False)[1:]]).T
				bin_edges_polar = np.array([np.linspace(0, 2*pi, nr_bins['polar'] + 1, endpoint = False)[:-1], np.linspace(0, 2*pi, nr_bins['polar'] + 1, endpoint = False)[1:]]).T
				# boolean arrays for positions in the visual field
				position_bins = np.array([[((eccen_data > eb[0]) * (eccen_data < eb[1]) * (polar_data > pb[0]) * (polar_data < pb[1])) for eb in bin_edges_eccen] for pb in bin_edges_polar], dtype = bool)
				# print position_bins.shape, eccen_data.shape, [[(position_bins[i,j,:,0] * nonzeros.T).sum() for j in range(nr_bins['eccen'])] for i in range(nr_bins['polar'])]
				for value in values:
					all_data = self.roi_data_from_hdf(h5file, run, roi, value)
					# print all_data.shape
					position_data = [[all_data[position_bins[i,j,:,0] * nonzeros, 0] for j in range(nr_bins['eccen'])] for i in range(nr_bins['polar'])]
					data[-1].append(position_data)
					# print position_data
		h5file.close()
		return data
	
	def take_retinotopic_data(self, rois = ['V1', 'V2', 'V3', 'V4', 'V3A'], values = ['alternation_cope'], nr_bins = {'eccen': 4, 'polar': 4}):
		all_data = []
		for r in [self.runList[i] for i in self.conditionDict['sphere_presto']]:
			all_data.append(self.take_retinotopic_data_from_run(run = r, rois = rois, values = values, nr_bins = nr_bins))
		all_data.append(self.take_retinotopic_data_from_run(run = r, rois = rois, values = [v+'_gfeat' for v in values], nr_bins = nr_bins, offsets = {'eccen': 0.0, 'polar': pi/2.0}))
		
		print [[all_data[-1][0][0][i][j].mean() for i in range(nr_bins['polar'])] for j in range(nr_bins['eccen'])]
		
		return all_data[-1]
	
	def new_state_decoding_roi(self, roi, data_type = 'psc_hpf_data', thresholds = [2.3, 30.3], mask_type = 'stim_on_Z'):
		h5file = self.hdf5_file()
		# mask data
		mask_data = np.array([self.roi_data_from_hdf(h5file, self.runList[i], roi, mask_type) for i in self.conditionDict['sphere_presto']]).mean(axis = 0)
		masks = ((mask_data > thresholds[0]) * (mask_data < thresholds[1])).ravel()
		
		these_data = []
		for r in [self.runList[i] for i in self.conditionDict['sphere_presto']]:
			these_data.append(self.roi_data_from_hdf(h5file, r, roi, data_type))
		# here's the chance to do some preprocessing on these data
		these_data = np.hstack(these_data)
		these_masked_data = [these_data[m,:].T for m in masks]
		
		# incorporate behavior
		percepts = np.vstack((self.allPercepts[:,0], self.allPercepts[:,1], self.allPercepts[:,2]-self.allPercepts[:,1])).T
		whichPercepts = percepts[:,0] == 66
		eventData = np.vstack([percepts[:,1], percepts[:,2], np.ones((percepts.shape[0]))]).T
		eventData = [eventData[whichPercepts], eventData[-whichPercepts]]
		
		from ..Operators.ImageOperator import Design
		
		d = Design(these_masked_data.shape[0], 0.65, subSamplingRatio = 100)
		# percept 1 regressor
		d.addRegressor(eventData[0])
		# percept 2 regressor
		d.addRegressor(eventData[1])
		
		d.convolveWithHRF(hrfType = 'singleGamma', hrfParameters = {'a': 6, 'b': 0.9}) 
		
		withinRunIndices = np.mod(np.arange(these_masked_data.shape[0]), self.timepointsPerRun) + ceil(4.0 / self.rtime)		
		whichSamplesAllRuns = (withinRunIndices > sampleInterval) * (withinRunIndices < (self.timepointsPerRun - sampleInterval))
		
		dM = d.designMatrix[whichSamplesAllRuns]
		rD = these_masked_data[whichSamplesAllRuns]
		
		# or not use median
		over_median = (dM[:,0] - dM[:,1]) > 0
		under_median = -over_median
		
		om_indices = np.arange(over_median.shape[0])[over_median]
		um_indices = np.arange(over_median.shape[0])[under_median]
		
		nr_runs = np.min([om_indices.shape[0], um_indices.shape[0]])
		dec = DecodingOperator(rD, decoder = 'svmLin', fullOutput = True)
		
		run_width = whichSamplesAllRuns.sum() / len(self.conditionDict['sphere_presto'])
		self.whichSamplesAllRuns = whichSamplesAllRuns
		
		decodingResultsArray = []
		for i in range(0, nr_runs-run_width, run_width):
			testThisRun = (np.arange(nr_runs) >= i) * (np.arange(nr_runs) < i+run_width)
			trainingThisRun = -testThisRun
			trainingDataIndices = np.concatenate(( om_indices[trainingThisRun], um_indices[trainingThisRun] ))
			testDataIndices = np.concatenate(( om_indices[testThisRun], um_indices[testThisRun] ))
			trainingsLabels = np.concatenate(( -np.ones((nr_runs-run_width)), np.ones((nr_runs-run_width)) ))
			testLabels = np.concatenate(( -np.ones((run_width)), np.ones((run_width)) ))
			
			res = dec.decode(trainingDataIndices, trainingsLabels, testDataIndices, testLabels)
			decodingResultsArray.append([testThisRun, res])
		
		allData = [self.allPercepts, d.designMatrix, whichSamplesAllRuns, decodingResultsArray]
		f = open(os.path.join(self.stageFolder(stage = 'processed/mri/sphere'), 'decodingResults_' + str(run_width) + '_' + roi + '.pickle'), 'wb')
		pickle.dump(allData, f)
		f.close()
	
	def stateDecodingFromRoi(self, roi, color = 'k', sampleInterval = 25, run_width = 1):
		self.logger.info('starting eventRelatedDecoding for roi %s', roi)
		
		roiData = self.gatherRIOData(roi, whichRuns = self.scanTypeDict['epi_bold'], whichMask = '_visual' )
		percepts = np.vstack((self.allPercepts[:,0], self.allPercepts[:,1], self.allPercepts[:,2]-self.allPercepts[:,1])).T
		whichPercepts = percepts[:,0] == 66
		eventData = np.vstack([percepts[:,1], percepts[:,2], np.ones((percepts.shape[0]))]).T
		eventData = [eventData[whichPercepts], eventData[-whichPercepts]]
		
		from ..Operators.ImageOperator import Design
		
		d = Design(roiData.shape[0], 0.65, subSamplingRatio = 100)
		# percept 1 regressor
		d.addRegressor(eventData[0])
		# percept 2 regressor
		d.addRegressor(eventData[1])
		
		d.convolveWithHRF(hrfType = 'singleGamma', hrfParameters = {'a': 6, 'b': 0.9}) 
		
		withinRunIndices = np.mod(np.arange(roiData.shape[0]), self.timepointsPerRun) + ceil(4.0 / self.rtime)		
		whichSamplesAllRuns = (withinRunIndices > sampleInterval) * (withinRunIndices < (self.timepointsPerRun - sampleInterval))
		
		dM = d.designMatrix[whichSamplesAllRuns]
		rD = roiData[whichSamplesAllRuns]
		
		from ..Operators.ArrayOperator import DecodingOperator
		
		# or not use median
		over_median = (dM[:,0] - dM[:,1]) > 0
		under_median = -over_median
		
		om_indices = np.arange(over_median.shape[0])[over_median]
		um_indices = np.arange(over_median.shape[0])[under_median]
		
		nr_runs = np.min([om_indices.shape[0], um_indices.shape[0]])
		dec = DecodingOperator(rD, decoder = 'svmLin', fullOutput = True)
		
		if run_width == 0:
			run_width = whichSamplesAllRuns.sum() / len(self.conditionDict['sphere'])
			self.whichSamplesAllRuns = whichSamplesAllRuns
		
		decodingResultsArray = []
		for i in range(0, nr_runs-run_width, run_width):
			testThisRun = (np.arange(nr_runs) >= i) * (np.arange(nr_runs) < i+run_width)
			trainingThisRun = -testThisRun
			trainingDataIndices = np.concatenate(( om_indices[trainingThisRun], um_indices[trainingThisRun] ))
			testDataIndices = np.concatenate(( om_indices[testThisRun], um_indices[testThisRun] ))
			trainingsLabels = np.concatenate(( -np.ones((nr_runs-run_width)), np.ones((nr_runs-run_width)) ))
			testLabels = np.concatenate(( -np.ones((run_width)), np.ones((run_width)) ))
			
			res = dec.decode(trainingDataIndices, trainingsLabels, testDataIndices, testLabels)
			decodingResultsArray.append([testThisRun, res])
		
		allData = [self.allPercepts, d.designMatrix, whichSamplesAllRuns, decodingResultsArray]
		f = open(os.path.join(self.stageFolder(stage = 'processed/mri/sphere'), 'decodingResults_' + str(run_width) + '_' + roi + '.pickle'), 'wb')
		pickle.dump(allData, f)
		f.close()
	
	def decodeEvents(self, run_width):
		areas = ['V1','V2','MT','pIPS','lh.precentral','superiorfrontal','inferiorparietal','superiorparietal']
		colors = np.linspace(0,1,len(areas))
#		fig = pl.figure(figsize = (4,12))
#		fig.subplots_adjust(wspace = 0.2, hspace = 0.4, left = 0.1, right = 0.9, bottom = 0.025, top = 0.975)
		for i in range(len(areas)):
#			s = fig.add_subplot(len(areas),1,i+1)
			self.stateDecodingFromRoi(areas[i], color = (colors[i],0,0), run_width = run_width)
#			s.set_title(areas[i])
#			s.set_ylim((-0.05,0.15))
#		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'era_multiple_areas.pdf'))
	
	def timeForDecodingResults(self, run_width = 1):
		areas = ['V1','V2','MT','lh.precentral','superiorfrontal','pIPS','inferiorparietal','superiorparietal']
		if run_width == 0:
			run_width = self.whichSamplesAllRuns.sum() / len(self.conditionDict['sphere'])
		fileNameList = [os.path.join(self.stageFolder(stage = 'processed/mri/sphere'), 'decodingResults_' + str(run_width) + '_' + area + '.pickle') for area in areas]
		# take behavior from the first file
		f = open(fileNameList[0])
		d = pickle.load(f)
		f.close()
		allPercepts = np.vstack(([[0,0.00,0.00]],d[0]+[0.0, 4.0, 4.0]))
		allTRTimes = np.arange(0, 0.65 * d[2].shape[0], 0.65) + 0.0001
		lP = np.array([allPercepts[allPercepts[:,1] < t][-1,1] for t in allTRTimes])
		lastPerceptForAllTRs = np.array([[t - allPercepts[allPercepts[:,1] < t][-1,1], t, allPercepts[allPercepts[:,1] < t][-1,2]-allPercepts[allPercepts[:,1] < t][-1,1]] for t in allTRTimes])
		
#		for i in range(lastPerceptForAllTRs.shape[0]):
#			print lastPerceptForAllTRs[i]
		
		dM = d[1][d[2]]
		
		# or not use median
		over_median = (dM[:,0] - dM[:,1]) > 0
		under_median = -over_median
		
		om_indices = np.arange(over_median.shape[0])[over_median]
		um_indices = np.arange(over_median.shape[0])[under_median]
		
		fig = pl.figure(figsize = (5,12))
		fig.subplots_adjust(wspace = 0.2, hspace = 0.4, left = 0.1, right = 0.9, bottom = 0.025, top = 0.975)
		
		# take decoding results
		for i in range(len(fileNameList)):
			f = open(fileNameList[i])
			d = pickle.load(f)
			f.close()
			
			decRes = d[3]
			
			weightsAndPerceptTime = []
			for singleTrial in decRes:
				whatTRs = np.concatenate([om_indices[singleTrial[0]], um_indices[singleTrial[0]]])
				TRs = lastPerceptForAllTRs[whatTRs]
				# we take the sign of the W vectors. 
				weights = (singleTrial[1][1] * (np.ones((singleTrial[1][1].shape[0] / 2, 2)) * [-1, 1]).T.ravel()) / 2.0 + 0.5
				allThisTRData = np.array([ TRs[:,0], TRs[:,0] / TRs[:,2], weights]).T
				allThisTRData = allThisTRData[(allThisTRData[:,0] != 0.0) * (allThisTRData[:,1] <= 1.0)]
				if allThisTRData.shape[0] > 0:
					weightsAndPerceptTime.append(allThisTRData)
			weightsAndPerceptTime = np.vstack(weightsAndPerceptTime)
			order = np.argsort(weightsAndPerceptTime[:,0])
			nrSamples = order.shape[0]
			nrBins = 5
			binEdgeIndices = np.array([np.round(np.linspace(0,nrSamples-1,nrBins+1))[:-1],np.round(np.linspace(0,nrSamples-1,nrBins+1))[1:]]).T
			binMiddles = np.linspace( 1.0 / (2*nrBins), 1.0 -  1.0 / (2*nrBins), nrBins)
			
			smooth_width = 200
			kern = stats.norm.pdf( np.linspace(-3.25,3.25,smooth_width) )
			kern = kern / kern.sum()
			sm_signal = np.convolve( weightsAndPerceptTime[order,2], kern, 'valid' )
			sm_time = np.convolve( weightsAndPerceptTime[order,0] , kern, 'valid' )
			sm_time_r = np.convolve( weightsAndPerceptTime[order,1] , kern, 'valid' )
			
			m_signal = np.array([[binMiddles[k], weightsAndPerceptTime[order[binEdgeIndices[k,0]:binEdgeIndices[k,1]], 2].mean(), weightsAndPerceptTime[order[binEdgeIndices[k,0]:binEdgeIndices[k,1]], 2].std()/sqrt(binEdgeIndices[k,1]-binEdgeIndices[k,0])] for k in range(nrBins)])
			
			print m_signal
			
			s = fig.add_subplot(len(areas),1,i+1)
	#		pl.plot( sm_time_r, sm_signal, 'r', alpha = 0.25, linewidth = 2.75 )
			pl.plot( m_signal[:,0], m_signal[:,1], 'r--', alpha = 0.85, linewidth = 2.75 )
			s.fill_between(m_signal[:,0], m_signal[:,1] - m_signal[:,2], m_signal[:,1] + m_signal[:,2], color = 'r', alpha = 0.25)
	#		pl.plot(weightsAndPerceptTime[:,1], weightsAndPerceptTime[:,2]/weightsAndPerceptTime[:,2].std(), 'ro', alpha = 0.1)
			s.axis([0,1,0.5,1])
		#	s.set_ylim([0,1])
			s.set_title(areas[i])
#3			s = s.twiny()
#			pl.plot( sm_time_r, sm_signal, color = 'r', alpha = 0.85, linewidth = 2.75 )
#			pl.plot(weightsAndPerceptTime[:,0], weightsAndPerceptTime[:,2], 'ro', alpha = 0.1)
#			s.set_ylim([0,1])
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'decoding_multiple_areas_' + str(run_width) + '.pdf'))
	
	
	def mapDecodingRoi(self, roiData, intervalForFit = [25,325], intervalForTest = [25,325], output_type = 'project', hemodynamic_lag_for_percepts = 4.0):
		percepts = np.vstack((self.allPercepts[:,0], self.allPercepts[:,1], self.allPercepts[:,2]-self.allPercepts[:,1])).T
		whichPercepts = percepts[:,0] == 66
		eventData = np.vstack([percepts[:,1], percepts[:,2], np.ones((percepts.shape[0]))]).T
		eventData = [eventData[whichPercepts], eventData[-whichPercepts]]
		# construct single regressor for both percepts
		eventData[1][:,-1] = -1
		eventData = np.concatenate((eventData[0],eventData[1]))
		eventData = eventData[np.argsort(eventData[:,0])]
		
		# construct where a TR falls inside a percept
		allTRTimes = np.arange(0, 0.65 * roiData.shape[0], 0.65) + 0.0001
		stimOffEvents = np.array([[-16.0 + i * (self.timepointsPerRun * self.rtime), 16, 0] for i in range(len(self.conditionDict['sphere'])+2)])
		
		allPerceptEventsForLastPerceptAnalysis = np.vstack((eventData, stimOffEvents))
		allPerceptEventsForLastPerceptAnalysis = allPerceptEventsForLastPerceptAnalysis[np.argsort(allPerceptEventsForLastPerceptAnalysis[:,0])]
#		import pdb; pdb.set_trace()
		lP = np.array([[
						allPerceptEventsForLastPerceptAnalysis[allPerceptEventsForLastPerceptAnalysis[:,0] < t][-1,-1], 
						t - allPerceptEventsForLastPerceptAnalysis[(allPerceptEventsForLastPerceptAnalysis[:,0] + hemodynamic_lag_for_percepts) < t][-1,0], 
						allPerceptEventsForLastPerceptAnalysis[(allPerceptEventsForLastPerceptAnalysis[:,0] + hemodynamic_lag_for_percepts) > t][0,0] - t
						] for t in allTRTimes])
		
		from ..Operators.ImageOperator import Design
		
		d = Design(roiData.shape[0], 0.65, subSamplingRatio = 100)
		# add percept regressor
		d.addRegressor(eventData)
		# transition events which we regress out - check what happens if we do or do not do this - well nothing since we're not interested in the T scores
		transitionEvents = np.vstack((percepts[:,0], percepts[:,1], np.ones(percepts.shape[0]) * 0.5)).T
		d.addRegressor(transitionEvents)
		# stimulus on events - these regressed out always
		stimulusOnEvents = [[1, 16.0 + i * (self.timepointsPerRun * self.rtime), (self.timepointsPerRun * self.rtime) - 16] for i in range(len(self.conditionDict['sphere']))]
		d.addRegressor(stimulusOnEvents)
		
		d.convolveWithHRF(hrfType = 'singleGamma', hrfParameters = {'a': 6, 'b': 0.9})
		
		# what timepoints to use for training and testing...
		withinRunIndices = np.mod(np.arange(roiData.shape[0]), self.timepointsPerRun) + ceil(4.0 / self.rtime)
		whichTrainSamplesAllRuns = (withinRunIndices > intervalForFit[0]) * (withinRunIndices < intervalForFit[1])
		whichTestSamplesAllRuns = (withinRunIndices > intervalForTest[0]) * (withinRunIndices < intervalForTest[1])
			
		# What percepts were dominant, used for later testing
		whatPercept = d.designMatrix[:,0] > 0
		wP = whatPercept[whichTestSamplesAllRuns]
		
		# this is the data that goes into the GLM
		dM = d.designMatrix[whichTrainSamplesAllRuns]
		rD = roiData[whichTrainSamplesAllRuns]
		lP = lP[whichTrainSamplesAllRuns]
		
		# GLM
		betas, sse, rank, sing = sp.linalg.lstsq( dM, rD, overwrite_a = True, overwrite_b = True )
		
		if output_type == 'corr':
			# spearman correlation 
			patternTimeCourses = np.array([spearmanr(t, betas[0])[0] for t in roiData[whichTestSamplesAllRuns]])
			perceptsDecoding = np.sign(wP * patternTimeCourses)
			accuracy = (1.0 + (perceptsDecoding.sum() / perceptsDecoding.shape[0])) / 2.0
			
		elif output_type == 'project':
			# projection
			patternTimeCourses = np.dot(betas[0] / np.sqrt(sse), roiData[whichTestSamplesAllRuns].T)
			perceptsDecoding = np.sign(wP * patternTimeCourses)
			accuracy = (1.0 + (perceptsDecoding.sum() / perceptsDecoding.shape[0])) / 2.0
			
			fig = pl.figure(figsize = (10,3))
			s = fig.add_subplot(111)
			pl.plot(patternTimeCourses, 'r')
			s = pl.twinx()
			pl.plot(wP-0.5, 'g')
			pl.plot(rD.mean(axis = 1)-rD.mean(), 'b')
			pl.draw()
			
		return {'patternTimeCourse': patternTimeCourses, 'perceptsDecoding':perceptsDecoding, 'accuracy':accuracy, 'betas':betas[-1], 'betas.mean':betas[-1].mean(), 'design_matrix':dM, 'perceptTimeForTR': lP, 'boldTimeCourse': rD.mean(axis = 1)}
	
	def mapDecoding(self, areas = ['V1',['V2v','V2d'],['V3v','V3d'],'V3A',['inferiorparietal','superiorparietal']], masks = ['_visual','_neg-visual']):
		def autolabel(rects):
			# attach some text labels
			for rect in rects:
				height = rect.get_height()
				pl.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
				ha='center', va='bottom')
		
		mapDR = []
		mapDNC = []
		for area in areas:
			mapDR.append([])
			mapDNC.append([])
			for mask in masks:
				a = presentSession.mapDecodingRoi(roiData = self.gatherRIOData(area, whichRuns = self.scanTypeDict['epi_bold'], whichMask = mask ))
				mapDR[-1].append(a[1])
				mapDNC[-1].append(a[3])
		mapDR = np.array(mapDR)
		mapDNC = np.array(mapDNC)
		fig = pl.figure()
		s = fig.add_subplot(2,1,1)
		rects1 = pl.bar(np.arange(len(areas)), mapDR[:,0], width, yerr = np.zeros(len(areas)), color='r', alpha = 0.4)
		rects2 = pl.bar(np.arange(len(areas))+width, mapDR[:,1], width, yerr = np.zeros(len(areas)), color='b', alpha = 0.4)
		
		pl.ylabel('Accuracy [\% correct]')
		pl.title('Decoding based on spearman\'s correlation with feature map from betas')
		pl.xticks(np.arange(len(areas))+width, areas )
		
		pl.legend( (rects1[0], rects2[0]), ('Visual', 'Negative Visual') )
		
		s.axis([-0.5,len(areas)-0.5, 0.1, 0.7])
		autolabel(rects1)
		autolabel(rects2)
		
		s = fig.add_subplot(2,1,2)
		rects1 = pl.bar(np.arange(len(areas)), mapDNC[:,0], width, yerr = np.zeros(len(areas)), color='r', alpha = 0.4)
		rects2 = pl.bar(np.arange(len(areas))+width, mapDNC[:,1], width, yerr = np.zeros(len(areas)), color='b', alpha = 0.4)
		
		# add some
		pl.ylabel('Accuracy [\% correct]')
		pl.title('Decoding based on projection of data on feature map from betas')
		pl.xticks(np.arange(len(areas))+width, areas )
		
		pl.legend( (rects1[0], rects2[0]), ('Visual', 'Negative Visual') )
		s.axis([-0.5,len(areas)-0.5, 0.45, 0.8])
		autolabel(rects1)
		autolabel(rects2)
	
	def eccenMapDecoding(self, areas = [['V1','V2v','V2d','V3v','V3d'],['V3A','V3B','V7?','MT'],['V4','LO1','LO2','fusiform','parahippocampal','inferiortemporal'],['superiorparietal','inferiorparietal','supramarginal','precuneus']]):#,'V3A',['inferiorparietal','superiorparietal'], ,'V3A','V4',['inferiortemporal','fusiform','parahippocampal'], 'V1',['V2v','V2d'],['V3v','V3d'],
	# ['V1',['V2v','V2d'],['V3v','V3d'],['V3A','V3B','V7?'],'V4',['LO1','LO2'],'MT',['superiorparietal','inferiorparietal','supramarginal','precuneus']]
		eccenFile = NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/') , 'eccen.nii.gz'))
		statFile = NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/') , 'visual.nii.gz'))
		
		allFuncData = []
		for r in self.conditionDict['sphere']:
			allFuncData.append(NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[r], postFix = ['mcf','Z'] )).data)
		allFuncData = np.vstack(allFuncData)
		
		colors = [(i/float(len(areas)), 1.0 - i/float(len(areas)), 0.0) for i in range(len(areas))]
		res = []
		fig = pl.figure(figsize = (7,10))
		plnr = 1
		patternTimeCourses = []
		allPhases = []
		allAcc = []
		for (i, a) in zip(range(len(areas)), areas):
			s = fig.add_subplot(len(areas),2,plnr)
			res.append(self.eccenMapDecodingFromRoi(eccenFile, statFile, allFuncData, s = s, area = a, nrBins = 15))
			s = pl.twinx()
			phases = np.array([r['phase'] for r in res[-1]])
			accuracy = np.array([r['accuracy'] for r in res[-1]])
			betas = np.array([r['betas.mean'] for r in res[-1]])
			patternTimeCourses.append(np.array([r['patternTimeCourse'] for r in res[-1]]))
			pl.plot(phases, accuracy, c = colors[0])
			s.set_ylabel('percentage correct', fontsize=9)
			s.set_xlim([0,2*pi])
			s.set_title(str(a))
			plnr += 1
			s = fig.add_subplot(len(areas),2,plnr)
			pl.plot(accuracy - accuracy.mean(), betas - betas.mean(), 'o', mfc = 'w', mec = 'k')
			pl.text(-0.02, 0, "$\rho$:%1.3f, p:%1.3f" % tuple(spearmanr(betas, accuracy)), fontsize=11)
			s.set_ylabel('beta')
			s.set_xlabel('perc correct')
			plnr += 1
			allPhases.append(phases)
			allAcc.append(accuracy)
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'eccen_phase_vs_stat_acc.pdf'))
		pl.draw()
		
		fig = pl.figure(figsize = (7,3))
		s = fig.add_subplot(111)
		for i in range(len(areas)):
			pl.plot(allPhases[i], allAcc[i], marker = '--', c = colors[i], label = str(areas[i]))
		s.axis([0,2*pi,0.50,0.65])
		leg = pl.legend()
		if leg:
			for t in leg.get_texts():
			    t.set_fontsize('small')    # the legend text fontsize
			for l in leg.get_lines():
			    l.set_linewidth(3.5)  # the legend line width
		
		pl.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'eccen_phase_vs_accuracy.pdf'))
		pl.draw()	
		return res
		
	
	def eccenMapDecodingFromRoi(self, eccenFile, statFile, allFuncData, s = None, area = ['V1'], nrBins = 15, binningType = 'nrVoxels'):
		if area.__class__.__name__ == 'str':
			area = [area]
		roiData = []
		for a in area:
			roiData.append( np.array([NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/masks/') , 'lh.' + a + '.nii.gz')).data, NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/masks/') , 'rh.' + a + '.nii.gz')).data]).sum(axis = 0) )
		roiData = np.array(np.array(roiData).sum(axis = 0), dtype = bool) * (eccenFile.data != 0.0)
		
		eccenRoiData = eccenFile.data[roiData]
		statRoiData = statFile.data[roiData]
		
		funcData = allFuncData[:,roiData]
		
		# check out the stats for different eccentricities
		# shift the phase back by 0.5 pi
		erd = np.fmod(eccenRoiData + 3.75 * pi, (2.0 * pi))
		ers = np.argsort(erd)
		erdT = np.concatenate((erd[ers]-(2*pi), erd[ers], erd[ers]+(2*pi)))
		srdT = np.tile(statRoiData[ers],3)
		
		pl.plot(erdT, srdT, 'ro', alpha = 0.25, ms = 4)
		
		smooth_width = 100
		kern = norm.pdf( np.linspace(-3.25,3.25,smooth_width) )
		kern = kern / kern.sum()
		sm_signal = np.convolve( srdT, kern, 'valid' )
		sm_ph = np.convolve( erdT, kern, 'valid' )
		pl.plot(sm_ph, sm_signal, 'r')
		pl.plot(np.linspace(0,2* pi,200), np.zeros((200)), 'k--')
		s.set_xlim([0,2*pi])
		s.set_xlabel('eccen phase, 0 = fovea', fontsize=9)
		s.set_ylabel('beta weight', fontsize=9)
		
		res = []
		if binningType == 'nrVoxels':	# if the bins have to be equally large
			nrVoxInBins = int(np.ceil(erd.shape[0] / float(nrBins)))
	#		nrVoxInBins = min(int(np.ceil(erd.shape[0] / float(nrBins))), 100)
	#		nrVoxInBins = 50
			for i in np.arange(0, erd.shape[0]-nrVoxInBins, nrVoxInBins):
				thisRes = self.mapDecodingRoi( roiData = funcData[:,ers[i:nrVoxInBins+i]], intervalForFit = [100,325], intervalForTest = [100,325])
				thisRes.update({'phase':np.mean(erd[ers[i:nrVoxInBins+i]])})
				res.append(thisRes)
		elif binningType == 'byPhase':
			binEdges = np.linspace(0,2*pi,nrBins+1)
			binEdges = np.vstack((binEdges[:-1], binEdges[1:])).T
			indices = [(erd > be[0]) * (erd < be[1]) for be in binEdges]
			for i in range(nrBins):
				thisRes = self.mapDecodingRoi( roiData = funcData[:,indices[i]], intervalForFit = [100,325], intervalForTest = [100,325])
				thisRes.update({'phase':np.mean(erd[indices[i]])})
				res.append(thisRes)
		return res
	
	def eccenMapDecodingCorr(self, areas = ['V1',['V2v','V2d'],['V3v','V3d'],['V3A','V3B','V7?'],['superiorparietal','inferiorparietal','supramarginal','precuneus']]):
		"""docstring for eccenMapDecodingCorr"""
		
		import nitime
		#Import the time-series objects:
		from nitime.timeseries import TimeSeries
		#Import the analysis objects:
		from nitime.analysis import CorrelationAnalyzer, CoherenceAnalyzer
		#Import utility functions:
		from nitime.utils import percent_change
		from nitime.viz import drawmatrix_channels, drawgraph_channels, plot_xcorr
		
		eccenFile = NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/') , 'eccen.nii.gz'))
		statFile = NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/masks/stat/') , 'visual.nii.gz'))
		
		allFuncData = []
		for r in self.conditionDict['sphere']:
			allFuncData.append(NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[r], postFix = ['mcf','Z'] )).data)
		allFuncData = np.vstack(allFuncData)
		
		nrBins = 8
		fig = pl.figure(figsize = (7,10))
		plnr = 1
		
		res = []
		patternTimeCourses = []
		boldTimeCourses = []
		shownames = ['V1','V2','V3','V3ab7','ant_par']
		names = []
		for (i, a) in zip(range(len(areas)), areas):
			s = fig.add_subplot(len(areas),1,plnr)
			res.append(self.eccenMapDecodingFromRoi(eccenFile, statFile, allFuncData, s = s, area = a, nrBins = nrBins, binningType = 'byPhase'))
			for j in range(nrBins):
				names.append(shownames[i] + '_' + str(j))
				patternTimeCourses.append(res[-1][j]['patternTimeCourse'])
				boldTimeCourses.append(res[-1][j]['boldTimeCourse'])
				trAndPerceptTimes = res[-1][j]['perceptTimeForTR']
			plnr += 1
		
		trPerceptTimeRatios = trAndPerceptTimes[:,1] / (trAndPerceptTimes[:,1] + trAndPerceptTimes[:,2])
		patternTimeCourses = np.array(patternTimeCourses)
		boldTimeCourses = np.array(boldTimeCourses)
		
		# analyse this without regard for time in percept
		if True:
			ts = TimeSeries(np.array(patternTimeCourses), sampling_interval = self.rtime)
			ts.metadata['roi'] = np.array(names)
		
			#Initialize the correlation analyzer
			C = CorrelationAnalyzer(ts)
		
			#Display the correlation matrix
			fig01 = drawmatrix_channels(C.corrcoef, np.array(names), size=[12., 12.], color_anchor=0)
			pl.savefig( os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'corr_pattern.pdf') )
			ts2 = TimeSeries(np.array(boldTimeCourses), sampling_interval = self.rtime)
			ts2.metadata['roi'] = np.array(names)
		
			#Initialize the correlation analyzer
			C2 = CorrelationAnalyzer(ts2)
		
			#Display the correlation matrix
			fig02 = drawmatrix_channels(C2.corrcoef, np.array(names), size=[12., 12.], color_anchor=0)
			pl.savefig( os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'corr_bold.pdf') )		
		
		# analyse this with regard for time in percept - time in seconds and time in total percept duration ratio
		if True:
			indicesTime, indicesRatio = np.argsort(trAndPerceptTimes[:,1]), np.argsort(trPerceptTimeRatios)
			nrTimeBins = 8
			nrTimesPerBin = int(floor(trAndPerceptTimes.shape[0] / float(nrTimeBins)))
			ratioResults = []
			for i in range(nrTimeBins):
				# per time
				ts = TimeSeries(patternTimeCourses[:,indicesTime[i * nrTimesPerBin:(i+1) * nrTimesPerBin]], sampling_interval = self.rtime)
				ts.metadata['roi'] = np.array(names)
				C = CorrelationAnalyzer(ts)
				#Display the correlation matrix
				fig01 = drawmatrix_channels(C.corrcoef, np.array(names), size=[12., 12.], color_anchor=0)
				pl.savefig( os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'corr_pattern_time_' + str(i) + '.pdf') )
				# per ratio
				ts = TimeSeries(patternTimeCourses[:,indicesRatio[i * nrTimesPerBin:(i+1) * nrTimesPerBin]], sampling_interval = self.rtime)
				ts.metadata['roi'] = np.array(names)
				C = CorrelationAnalyzer(ts)
				#Display the correlation matrix
				fig01 = drawmatrix_channels(C.corrcoef, np.array(names), size=[12., 12.], color_anchor=0)
				ratioResults.append(C.corrcoef)
				pl.savefig( os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'corr_pattern_ratio_' + str(i) + '.pdf') )
		ratioResults = np.array(ratioResults)	# time, areacombinations
		corrOverTimeBetweenAreas = np.array([[spearmanr(range(nrTimeBins), r)[0] for r in rr] for rr in np.transpose(ratioResults,(1,2,0))])
		corrOverTimeBetweenAreasPVal = np.array([[spearmanr(range(nrTimeBins), r)[1] for r in rr] for rr in np.transpose(ratioResults,(1,2,0))])
		whatPVals = corrOverTimeBetweenAreasPVal > 0.05
		corrOverTimeBetweenAreas[whatPVals] = 0.0
		print corrOverTimeBetweenAreas.shape, C.corrcoef.shape
		fig01 = drawmatrix_channels(corrOverTimeBetweenAreas, np.array(names), size=[12., 12.], color_anchor=0)
		pl.savefig( os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'corr_pattern_corr_over_time_ratio.pdf') )
#		import pdb; pdb.set_trace()
	