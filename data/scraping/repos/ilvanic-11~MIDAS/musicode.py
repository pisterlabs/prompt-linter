# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------
# Name:         musicode.py
# Purpose:      This is the pianoroll_mainbuttons_split file for musicode
#
# Authors:      Zachary Plovanic - Lead Programmer
#               Isaac Plovanic - Creator, Director, Programmer, Editor, Administrator,
#
# Copyright:    MIDAS is Copyright © 2017-2019 Isaac Plovanic and Zachary Plovanic
#               music21 is Copyright © 2006-19 Michael Scott Cuthbert and the music21
#               Project
# License:      LGPL or BSD, see license.txt
#------------------------------------------------------------------------------------



###############################################################################
# TABLE OF CONTENTS
#
#MUSICODE_FUNCTIONS
#-------------------
#UA-1.  def TRANSLATE(musicode, string)
#UA-2.  def TRANSLATE_FROM_TEXT_FILE(musicode, filename)
#UA-3.  def TRANSLATE_LETTER(c, musicode, num) ???
#UA-4.  def TRANSLATE_EACH_LETTER_TO_RANDOM_MUSICODE(text)
#UA-5.  def ALIGN_MUSICODE_WITH_MELODY(melody_stream, musicode)
#TODO UA-6.   def CHANGE_MUSICODE

###############################################################################

import copy
import music21
import cv2
import numpy as np
import re
import os
import random

import openai

#from midas_scripts import music21funcs
import midas_scripts
from midas_scripts import music21funcs
from midas_scripts.midiart import cv2_tuple_reconversion, make_midi_from_grayscale_pixels
from collections import OrderedDict
musicode_reference = music21.converter.thaw(r"C:\Users\Isaac's\Midas\resources\intermediary_path\musicode_test.p")
openai.api_key = ''.join([i[1].obj[-1] for i in musicode_reference])
from traits.api import HasTraits
from traits.trait_types import Any
import shutil


class Musicode():

	def __init__(self):
		self.user_Created = music21.stream.Part()
		self.musicode_name = "User_Generated"
		self.sh = "ug"
		self.full_new_musicode_path = ""  #This is established when a new musicode is created with make_musicode(). #TODO Make a default? Or leave so export throws an error?
		#print("userCreated on Init:", self.userCreated)
		## The function "make_musicode" appends to this and allows the user to translate from it and\or export\write from it.
		# An instantiation of musicode.Musicode() occurs in the MIDAS_wx.py init.

		# Establish operating data for the make_musicode() function.
		self.Latin_Script = '''AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz  ?,;\':-.!\"()[]/  0123456789'''
		self.Punct_Workaround = OrderedDict.fromkeys(([j for j in '''?,;\':-.!\"()[]/''']))
		self.Punct_Symbols = '''?,;\':-.!\"()[]/'''
		self.Punct_Names = ['questionmark', 'comma', 'semicolon', 'singlequotationmark', 'colon', 'hyphen', 'period',
							'exclamationmark',
							'doublequotationmark', 'openparenthesis', 'closeparenthesis', 'openbracket', 'closebracket',
							'forwardslash']
		for l in range(0, len(self.Punct_Symbols)):
			self.Punct_Workaround[self.Punct_Symbols[l]] = self.Punct_Names[l]



	#self.SetupDefaultMidiDictionaries()
		#self.setup_default_musicode_dictionaries()

	set_path = r"musicode_libraries"
	absFilePath = os.path.dirname(os.path.abspath(set_path))
	libraryPath = absFilePath + "\\resources\\" + set_path
	musicode_path = libraryPath
	# The path to the top level of Musicode midi Libraries
	#musicode_path = r"C:\Users\Zach-X\PycharmProjects\MIDAS\resources\musicode_libraries"
	#musicode_path = r"C:\Users\Isaac's\Desktop\Isaacs_Synth_Music_Source_Folder\FL\Workflow\10_Musicode_Libraries"  #TODO Needs to be localized to installation folders.
	#musicode_path = r"C:\Users\iplovanic\Desktop\10_Musicode Libraries"
	#musicode_path = r"C:\Users\iplovanic\Desktop\MIDAS\10_Musicode_Libraries"

	am_dict = dict() #Animuse dictionary
	asciiX_dict = dict()  #Asciipher dictionary
	asciiY_dict = dict()  #Asciipher dictionary
	bp_dict = dict() #BraillePulse dictionary
	mm_dict = dict() #Metamorse dictionary
	ptX_dict = dict() #POWerTap_X dictionary
	ptY_dict = dict() #POWerTap_Y dictionary
	se_dict = dict() #Script-Ease dictionary
	splyce_dict = dict() #Splyce dictionary
	boX_dict = dict() # Baud-Onkadonk dictionary
	boY_dict = dict() # Baud-Onkadonk dictionary

	shorthand = {
				 "Animuse"        : "am",
				 "Asciipher_X"    : "asciiX",
				 "Asciipher_Y"    : "asciiY",
				 "BraillePulse"   : "bp",
				 "MetaMorse"      : "mm",
				 "POWerTap_X"     : "ptX",
				 "POWerTap_Y"     : "ptY",
				 "Script-Ease"    : "se",
				 "Splyce"         : "splyce",
				 "Baud-Onkadonk_X": "boX",
				 "Baud-Onkadonk_Y": "boY",
				 }

	dictionaries = {
				 "Animuse"        : am_dict,
				 "Asciipher_X"    : asciiX_dict,
				 "Asciipher_Y"    : asciiY_dict,
				 "BraillePulse"   : bp_dict,
				 "MetaMorse"      : mm_dict,
				 "POWerTap_X"     : ptX_dict,
				 "POWerTap_Y"     : ptY_dict,
				 "Script-Ease"    : se_dict,
				 "Splyce"         : splyce_dict,
				 "Baud-Onkadonk_X" : boX_dict,
				 "Baud-Onkadonk_Y" : boY_dict,

				}

	setup_pairs = {"Animuse"           	: 'self._setup_am_dict',
					"Asciipher_X"      	: 'self._setup_se_dict',
					"Asciipher_Y"      	: 'self._setup_asciiX_dict',
					"BraillePulse"     	: 'self._setup_asciiY_dict',
					"MetaMorse"        	: 'self._setup_bp_dict',
					"POWerTap_X"       	: 'self._setup_mm_dict',
					"POWerTap_Y"       	: 'self._setup_ptX_dict',
					"Script-Ease"      	: 'self._setup_ptY_dict',
					"Splyce"           	: 'self._setup_splyce_dict',
					"Baud-Onkadonk_X"   : 'self._setup_boX_dict',
					"Baud-Onkadonk_Y"	: 'self._setup_boY_dict'}


	# def setup_default_musicode_dictionaries(self):
	# 	"""
	#
	# 	:return:
	# 	"""
	# 	#intermediary_path = os.getcwd() + os.sep + "resources" + os.sep + "intermediary_path" + os.sep
	# 	print(self.Latin_Script)
	# 	directory = os.getcwd() + os.sep + "resources" + os.sep + "musicode_libraries" + os.sep + "Musicode_Defaults" + os.sep
	# 	directory2 = r"C:\Users\Isaac's\Midas\resources\musicode_libraries\New_Musicodes"
	# 	for midifile in os.listdir(directory):
	# 		print(os.path.join(directory, midifile))
	# 		#if midifile.endswith(".mid"):
	# 		print(midifile)
	# 		parsed_midi = music21.converter.parse(directory + midifile)
	# 		#inPlace = True returns a None-Type
	# 		parsed_midi2 = parsed_midi.makeMeasures(inPlace=False)
	# 		#print("PARSED_MIDI HERE")
	# 		#parsed_midi.show('txt')
	# 		#For each one..
	# 		self.made_musicode = self.make_musicode(parsed_midi2, os.path.splitext(midifile)[0], self.shorthand[os.path.splitext(midifile)[0]], filepath=directory2, write=False, timeSig="4/4")     #TODO Set this later to user defined timeSig.
	# 		self.dictionaries[os.path.splitext(midifile)[0]] = self.made_musicode[0]
	#
	# 	self.user_Created = music21.stream.Part()
	# 	self.musicode_name = "User_Generated"


	def add_to_dict(self, char, musicode, full_path_to_file):
		"""
			This function is made public in case someone wants to modify the dicts.
		:param dict:
		:param full_path_to_file:
		:return:
		"""
		#print("Importing: %s" % full_path_to_file)
		m = music21.stream.Measure()
		w = music21.ElementWrapper("".join([musicode, " : ", char]))
		te = music21.expressions.TextExpression(char)
		te.style.fontSize = 18
		te.style.absoluteY = 40
		te.style.alignHorizontal = 'center'

		m.append(w)
		m.append(te)
		m.duration = music21.duration.Duration(4.0)
		imported_mid = music21.converter.parse(full_path_to_file)
		si = imported_mid.flat.notesAndRests
		#insert notes into new measure
		for el in si:
			m.insert(el)
		#insert rest to fill up remainder of measure
		if (m.highestTime < 4):
			rest = music21.note.Rest(quarterLength=(4-m.highestTime))
			m.append(rest)
		self.dictionaries[musicode][char] = m

	#TODO Rename this?
	def SetupDefaultMidiDictionaries(self, wx_progress_updater=None):

		from time import sleep #TODO #Consider revising
		methods_list = [self._setup_am_dict,
						self._setup_asciiX_dict,
						self._setup_asciiY_dict,
						self._setup_bp_dict,
						self._setup_mm_dict,
						self._setup_ptX_dict,
						self._setup_ptY_dict,
						self._setup_splyce_dict,
						self._setup_se_dict,
						self._setup_boX_dict,
						self._setup_boY_dict]

		i = 1
		for j in range(0, 11, 1):
			methods_list[j]()
			sleep(.25)
			try:
				#Original ProgressDialog
				#wx_progress_updater.Update(i)   #[i, 0], 50)

				#PyGauge
				#wx_progress_updater.Update(i, 1)   #[i, 0], 50)
				#wx_progress_updater.SetValue(i-.75)   #[i, 0], 50)
				#sleep(.45)
				wx_progress_updater.SetValue(i-.5)   #[i, 0], 50)
				sleep(.25)
				#wx_progress_updater.SetValue(i-.25)   #[i, 0], 50)
				#wx_progress_updater.Refresh()
				#sleep(.45) #0.05
				wx_progress_updater.SetValue(i)
				#sleep(.45)
				#print("VALUE", i)
			except Exception as e:
				#print("Musicode setup error; trying gauge....", e)

				#wx.Gauge
				wx_progress_updater.SetValue(i)

				#print("VALUE", i)
			i+=1
		# self._setup_am_dict()
		#
		# wx_progress_updater.Update(i)
		# i += 1
		#
		# self._setup_asciiX_dict()
		# wx_progress_updater.Update(i)
		# i += 1
		#
		# self._setup_asciiY_dict()
		# wx_progress_updater.Update(i)
		# i += 1
		#
		# self._setup_bp_dict()
		# wx_progress_updater.Update(i)
		# i += 1
		#
		# self._setup_mm_dict()
		# wx_progress_updater.Update(i)
		# i += 1
		#
		# self._setup_ptX_dict()
		# wx_progress_updater.Update(i)
		# i += 1
		#
		# self._setup_ptY_dict()
		# wx_progress_updater.Update(i)
		# i += 1
		#
		# self._setup_splyce_dict()
		# wx_progress_updater.Update(i)
		# i += 1
		#
		# self._setup_se_dict()
		# wx_progress_updater.Update(i)
		# i += 1
		#
		# self._setup_boX_dict()
		# wx_progress_updater.Update(i)
		# i += 1
		#
		# self._setup_boY_dict()
		# wx_progress_updater.Update(i)
		# i += 1

	def _setup_letters_and_numbers(self, musicode):
		sh = self.shorthand.get(musicode)
		mdict = self.dictionaries[musicode]
		for pack in ("Lowercase", "Uppercase", "Numbers"):
			directory = self.musicode_path + "\\" + musicode + "\\" + sh + "_" + pack
			filenames = next(os.walk(directory))[2]
			for file in filenames:
				my_search = re.search(r"musicode_%s_(\w|\number).mid"% sh, file)
				if my_search:
					self.add_to_dict(my_search.group(1), musicode, directory+"\\"+file)
		# Add a measure of rest for the space character (' ')
		r = music21.stream.Measure()
		r.duration = music21.duration.Duration(4.0)
		rest = music21.note.Rest(quarterLength=4.0)
		r.insert(rest)
		mdict[' '] = r
	#end _setup_letters_and_numbers()




	def _setup_am_dict(self): # punctuation_only=False):
		self._setup_letters_and_numbers("Animuse") #if punctuation_only is False else None
		punct_dir = self.musicode_path + "\\Animuse\\am_Punctuation\\"
		self.add_to_dict("/", "Animuse", "".join([punct_dir, "musicode_am_forwardslash.mid"]))
		self.add_to_dict("]", "Animuse", "".join([punct_dir, "musicode_am_closebracket.mid"]))
		self.add_to_dict(")", "Animuse", "".join([punct_dir, "musicode_am_closeparenthesis.mid"]))
		self.add_to_dict(":", "Animuse", "".join([punct_dir, "musicode_am_colon.mid"]))
		self.add_to_dict(",", "Animuse", "".join([punct_dir, "musicode_am_comma.mid"]))
		self.add_to_dict('\"', "Animuse", "".join([punct_dir, "musicode_am_doublequotationmark.mid"]))
		self.add_to_dict("!", "Animuse", "".join([punct_dir, "musicode_am_exclamationmark.mid"]))
		self.add_to_dict("-", "Animuse", "".join([punct_dir, "musicode_am_hyphen.mid"]))
		self.add_to_dict("[", "Animuse", "".join([punct_dir, "musicode_am_openbracket.mid"]))
		self.add_to_dict("(", "Animuse", "".join([punct_dir, "musicode_am_openparenthesis.mid"]))
		self.add_to_dict(".", "Animuse", "".join([punct_dir, "musicode_am_period.mid"]))
		self.add_to_dict(";", "Animuse", "".join([punct_dir, "musicode_am_semicolon.mid"]))
		self.add_to_dict("\'", "Animuse", "".join([punct_dir, "musicode_am_singlequotationmark.mid"]))
		self.add_to_dict("?", "Animuse", "".join([punct_dir, "musicode_am_questionmark.mid"]))
		print("Dictionary(s) established.")


	def _setup_se_dict(self): # punctuation_only=False):
		self._setup_letters_and_numbers("Script-Ease") #if punctuation_only is False else None
		punct_dir = self.musicode_path + "\\Script-Ease\\se_Punctuation\\"
		self.add_to_dict("/", "Script-Ease", "".join([punct_dir, "musicode_se_forwardslash.mid"]))
		self.add_to_dict("]", "Script-Ease", "".join([punct_dir, "musicode_se_closebracket.mid"]))
		self.add_to_dict(")", "Script-Ease", "".join([punct_dir, "musicode_se_closeparenthesis.mid"]))
		self.add_to_dict(":", "Script-Ease", "".join([punct_dir, "musicode_se_colon.mid"]))
		self.add_to_dict(",", "Script-Ease", "".join([punct_dir, "musicode_se_comma.mid"]))
		self.add_to_dict('\"', "Script-Ease", "".join([punct_dir, "musicode_se_doublequotationmark.mid"]))
		self.add_to_dict("!", "Script-Ease", "".join([punct_dir, "musicode_se_exclamationmark.mid"]))
		self.add_to_dict("-", "Script-Ease", "".join([punct_dir, "musicode_se_hyphen.mid"]))
		self.add_to_dict("[", "Script-Ease", "".join([punct_dir, "musicode_se_openbracket.mid"]))
		self.add_to_dict("(", "Script-Ease", "".join([punct_dir, "musicode_se_openparenthesis.mid"]))
		self.add_to_dict(".", "Script-Ease", "".join([punct_dir, "musicode_se_period.mid"]))
		self.add_to_dict(";", "Script-Ease", "".join([punct_dir, "musicode_se_semicolon.mid"]))
		self.add_to_dict("\'", "Script-Ease", "".join([punct_dir, "musicode_se_singlequotationmark.mid"]))
		self.add_to_dict("?", "Script-Ease", "".join([punct_dir, "musicode_se_questionmark.mid"]))
		print("Dictionary(s) established.")


	def _setup_asciiX_dict(self): # punctuation_only=False):
		self._setup_letters_and_numbers("Asciipher_X") #if punctuation_only is False else None
		punct_dir = self.musicode_path + "\\Asciipher_X\\asciiX_Punctuation\\"
		self.add_to_dict("\'", "Asciipher_X", "".join([punct_dir, "musicode_asciiX_singlequotationmark.mid"]))
		self.add_to_dict("/", "Asciipher_X", "".join([punct_dir, "musicode_asciiX_forwardslash.mid"]))
		self.add_to_dict("]", "Asciipher_X", "".join([punct_dir, "musicode_asciiX_closebracket.mid"]))
		self.add_to_dict(")", "Asciipher_X", "".join([punct_dir, "musicode_asciiX_closeparenthesis.mid"]))
		self.add_to_dict(":", "Asciipher_X", "".join([punct_dir, "musicode_asciiX_colon.mid"]))
		self.add_to_dict(",", "Asciipher_X", "".join([punct_dir, "musicode_asciiX_comma.mid"]))
		self.add_to_dict("!", "Asciipher_X", "".join([punct_dir, "musicode_asciiX_exclamationmark.mid"]))
		self.add_to_dict("-", "Asciipher_X", "".join([punct_dir, "musicode_asciiX_hyphen.mid"]))
		self.add_to_dict("[", "Asciipher_X", "".join([punct_dir, "musicode_asciiX_openbracket.mid"]))
		self.add_to_dict("(", "Asciipher_X", "".join([punct_dir, "musicode_asciiX_openparenthesis.mid"]))
		self.add_to_dict('\"', "Asciipher_X", "".join([punct_dir, "musicode_asciiX_doublequotationmark.mid"]))
		self.add_to_dict(".", "Asciipher_X", "".join([punct_dir, "musicode_asciiX_period.mid"]))
		self.add_to_dict(";", "Asciipher_X", "".join([punct_dir, "musicode_asciiX_semicolon.mid"]))
		self.add_to_dict("?", "Asciipher_X", "".join([punct_dir, "musicode_asciiX_questionmark.mid"]))# HDXF TGHS,M
		print("Dictionary(s) established.")


	def _setup_asciiY_dict(self): # punctuation_only=False):
		self._setup_letters_and_numbers("Asciipher_Y") #if punctuation_only is False else None
		punct_dir = self.musicode_path + "\\Asciipher_Y\\asciiY_Punctuation\\"
		self.add_to_dict("\'", "Asciipher_Y", "".join([punct_dir, "musicode_asciiY_singlequotationmark.mid"]))
		self.add_to_dict("/", "Asciipher_Y", "".join([punct_dir, "musicode_asciiY_forwardslash.mid"]))
		self.add_to_dict("]", "Asciipher_Y", "".join([punct_dir, "musicode_asciiY_closebracket.mid"]))
		self.add_to_dict(")", "Asciipher_Y", "".join([punct_dir, "musicode_asciiY_closeparenthesis.mid"]))
		self.add_to_dict(":", "Asciipher_Y", "".join([punct_dir, "musicode_asciiY_colon.mid"]))
		self.add_to_dict(",", "Asciipher_Y", "".join([punct_dir, "musicode_asciiY_comma.mid"]))
		self.add_to_dict("!", "Asciipher_Y", "".join([punct_dir, "musicode_asciiY_exclamationmark.mid"]))
		self.add_to_dict("-", "Asciipher_Y", "".join([punct_dir, "musicode_asciiY_hyphen.mid"]))
		self.add_to_dict("[", "Asciipher_Y", "".join([punct_dir, "musicode_asciiY_openbracket.mid"]))
		self.add_to_dict("(", "Asciipher_Y", "".join([punct_dir, "musicode_asciiY_openparenthesis.mid"]))
		self.add_to_dict('\"', "Asciipher_Y", "".join([punct_dir, "musicode_asciiY_doublequotationmark.mid"]))
		self.add_to_dict(".", "Asciipher_Y", "".join([punct_dir, "musicode_asciiY_period.mid"]))
		self.add_to_dict(";", "Asciipher_Y", "".join([punct_dir, "musicode_asciiY_semicolon.mid"]))
		self.add_to_dict("?", "Asciipher_Y", "".join([punct_dir, "musicode_asciiY_questionmark.mid"]))
		print("Dictionary(s) established.")


	def _setup_bp_dict(self): # punctuation_only=False):
		self._setup_letters_and_numbers("BraillePulse") #if punctuation_only is False else None
		punct_dir = self.musicode_path + "\\BraillePulse\\bp_Punctuation\\"
		self.add_to_dict("\'", "BraillePulse", "".join([punct_dir, "musicode_bp_singlequotationmark.mid"]))
		self.add_to_dict("/", "BraillePulse", "".join([punct_dir, "musicode_bp_forwardslash.mid"]))
		self.add_to_dict(":", "BraillePulse", "".join([punct_dir, "musicode_bp_colon.mid"]))
		self.add_to_dict(",", "BraillePulse", "".join([punct_dir, "musicode_bp_comma.mid"]))
		self.add_to_dict(".", "BraillePulse", "".join([punct_dir, "musicode_bp_decimalpoint.mid"]))
		self.add_to_dict("!", "BraillePulse", "".join([punct_dir, "musicode_bp_exclamationmark.mid"]))
		self.add_to_dict("-", "BraillePulse", "".join([punct_dir, "musicode_bp_hyphen.mid"]))
		self.add_to_dict("(", "BraillePulse", "".join([punct_dir, "musicode_bp_openclosebracketparenthesis.mid"]))
		self.add_to_dict(")", "BraillePulse", "".join([punct_dir, "musicode_bp_openclosebracketparenthesis.mid"]))
		self.add_to_dict("[", "BraillePulse", "".join([punct_dir, "musicode_bp_openclosebracketparenthesis.mid"]))
		self.add_to_dict("]", "BraillePulse", "".join([punct_dir, "musicode_bp_openclosebracketparenthesis.mid"]))
		self.add_to_dict('\"', "BraillePulse", "".join([punct_dir, "musicode_bp_doublequotationmark.mid"]))
		self.add_to_dict(".", "BraillePulse", "".join([punct_dir, "musicode_bp_period.mid"]))
		self.add_to_dict(";", "BraillePulse", "".join([punct_dir, "musicode_bp_semicolon.mid"]))
		self.add_to_dict("?", "BraillePulse", "".join([punct_dir, "musicode_bp_questionmark.mid"]))
		print("Dictionary(s) established.")


	def _setup_mm_dict(self): # punctuation_only=False):
		self._setup_letters_and_numbers("MetaMorse") #if punctuation_only is False else None
		punct_dir = self.musicode_path + "\\MetaMorse\\mm_Punctuation\\"
		self.add_to_dict("/", "MetaMorse", "".join([punct_dir, "musicode_mm_forwardslash.mid"]))
		self.add_to_dict("]", "MetaMorse", "".join([punct_dir, "musicode_mm_closebracketparenthesis.mid"]))
		self.add_to_dict(")", "MetaMorse", "".join([punct_dir, "musicode_mm_closebracketparenthesis.mid"]))
		self.add_to_dict(":", "MetaMorse", "".join([punct_dir, "musicode_mm_colon.mid"]))
		self.add_to_dict(",", "MetaMorse", "".join([punct_dir, "musicode_mm_comma.mid"]))
		self.add_to_dict('\"', "MetaMorse", "".join([punct_dir, "musicode_mm_doublequotationsmark.mid"]))
		self.add_to_dict("!", "MetaMorse", "".join([punct_dir, "musicode_mm_exclamationmark.mid"]))
		self.add_to_dict("-", "MetaMorse", "".join([punct_dir, "musicode_mm_hyphen.mid"]))
		self.add_to_dict("[", "MetaMorse", "".join([punct_dir, "musicode_mm_openbracketparenthesis.mid"]))
		self.add_to_dict("(", "MetaMorse", "".join([punct_dir, "musicode_mm_openbracketparenthesis.mid"]))
		self.add_to_dict(".", "MetaMorse", "".join([punct_dir, "musicode_mm_period.mid"]))
		self.add_to_dict(";", "MetaMorse", "".join([punct_dir, "musicode_mm_semicolon.mid"]))
		self.add_to_dict("\'", "MetaMorse", "".join([punct_dir, "musicode_mm_singlequotationmark.mid"]))
		self.add_to_dict("?", "MetaMorse", "".join([punct_dir, "musicode_mm_questionmark.mid"]))
		print("Dictionary(s) established.")


	def _setup_ptX_dict(self): # punctuation_only=False):
		self._setup_letters_and_numbers("POWerTap_X") #if punctuation_only is False else None
		punct_dir = self.musicode_path + "\\POWerTap_X\\ptX_Punctuation\\"
		self.add_to_dict("/", "POWerTap_X", "".join([punct_dir, "musicode_ptX_forwardslash.mid"]))
		self.add_to_dict(":", "POWerTap_X", "".join([punct_dir, "musicode_ptX_colon.mid"]))
		self.add_to_dict(",", "POWerTap_X", "".join([punct_dir, "musicode_ptX_comma.mid"]))
		self.add_to_dict('\"', "POWerTap_X", "".join([punct_dir, "musicode_ptX_doublequotationmark.mid"]))
		self.add_to_dict("!", "POWerTap_X", "".join([punct_dir, "musicode_ptX_exclamationmark.mid"]))
		self.add_to_dict("-", "POWerTap_X", "".join([punct_dir, "musicode_ptX_hyphen.mid"]))
		self.add_to_dict("[", "POWerTap_X", "".join([punct_dir, "musicode_ptX_openclosebracket.mid"]))
		self.add_to_dict("]", "POWerTap_X", "".join([punct_dir, "musicode_ptX_openclosebracket.mid"]))
		self.add_to_dict("(", "POWerTap_X", "".join([punct_dir, "musicode_ptX_opencloseparenthesis.mid"]))
		self.add_to_dict(")", "POWerTap_X", "".join([punct_dir, "musicode_ptX_opencloseparenthesis.mid"]))
		self.add_to_dict(".", "POWerTap_X", "".join([punct_dir, "musicode_ptX_period.mid"]))
		self.add_to_dict(";", "POWerTap_X", "".join([punct_dir, "musicode_ptX_semicolon.mid"]))
		self.add_to_dict("\'", "POWerTap_X", "".join([punct_dir, "musicode_ptX_singlequotationmark.mid"]))
		self.add_to_dict("?", "POWerTap_X", "".join([punct_dir, "musicode_ptX_questionmark.mid"]))
		print("Dictionary(s) established.")


	def _setup_ptY_dict(self): # punctuation_only=False):
		self._setup_letters_and_numbers("POWerTap_Y") #if punctuation_only is False else None
		punct_dir = self.musicode_path + "\\POWerTap_Y\\ptY_Punctuation\\"
		self.add_to_dict("/", "POWerTap_Y", "".join([punct_dir, "musicode_ptY_forwardslash.mid"]))
		self.add_to_dict(":", "POWerTap_Y", "".join([punct_dir, "musicode_ptY_colon.mid"]))
		self.add_to_dict(",", "POWerTap_Y", "".join([punct_dir, "musicode_ptY_comma.mid"]))
		self.add_to_dict('\"', "POWerTap_Y", "".join([punct_dir, "musicode_ptY_doublequotationmark.mid"]))
		self.add_to_dict("!", "POWerTap_Y", "".join([punct_dir, "musicode_ptY_exclamationmark.mid"]))
		self.add_to_dict("-", "POWerTap_Y", "".join([punct_dir, "musicode_ptY_hyphen.mid"]))
		self.add_to_dict("[", "POWerTap_Y", "".join([punct_dir, "musicode_ptY_openclosebracket.mid"]))
		self.add_to_dict("]", "POWerTap_Y", "".join([punct_dir, "musicode_ptY_openclosebracket.mid"]))
		self.add_to_dict("(", "POWerTap_Y", "".join([punct_dir, "musicode_ptY_opencloseparenthesis.mid"]))
		self.add_to_dict(")", "POWerTap_Y", "".join([punct_dir, "musicode_ptY_opencloseparenthesis.mid"]))
		self.add_to_dict(".", "POWerTap_Y", "".join([punct_dir, "musicode_ptY_period.mid"]))
		self.add_to_dict(";", "POWerTap_Y", "".join([punct_dir, "musicode_ptY_semicolon.mid"]))
		self.add_to_dict("\'", "POWerTap_Y", "".join([punct_dir, "musicode_ptY_singlequotationmark.mid"]))
		self.add_to_dict("?", "POWerTap_Y", "".join([punct_dir, "musicode_ptY_questionmark.mid"]))
		print("Dictionary(s) established.")


	def _setup_splyce_dict(self): # punctuation_only=False):
		self._setup_letters_and_numbers("Splyce") #if punctuation_only is False else None
		punct_dir = self.musicode_path + "\\Splyce\\splyce_Punctuation\\"
		self.add_to_dict("/", "Splyce", "".join([punct_dir, "musicode_splyce_forwardslash.mid"]))
		self.add_to_dict("]", "Splyce", "".join([punct_dir, "musicode_splyce_closebracket.mid"]))
		self.add_to_dict(")", "Splyce", "".join([punct_dir, "musicode_splyce_closeparenthesis.mid"]))
		self.add_to_dict(":", "Splyce", "".join([punct_dir, "musicode_splyce_colon.mid"]))
		self.add_to_dict(",", "Splyce", "".join([punct_dir, "musicode_splyce_comma.mid"]))
		self.add_to_dict('\"', "Splyce", "".join([punct_dir, "musicode_splyce_doublequotationmark.mid"]))
		self.add_to_dict("!", "Splyce", "".join([punct_dir, "musicode_splyce_exclamationmark.mid"]))
		self.add_to_dict("-", "Splyce", "".join([punct_dir, "musicode_splyce_hyphen.mid"]))
		self.add_to_dict("[", "Splyce", "".join([punct_dir, "musicode_splyce_openbracket.mid"]))
		self.add_to_dict("(", "Splyce", "".join([punct_dir, "musicode_splyce_openparenthesis.mid"]))
		self.add_to_dict(".", "Splyce", "".join([punct_dir, "musicode_splyce_period.mid"]))
		self.add_to_dict(";", "Splyce", "".join([punct_dir, "musicode_splyce_semicolon.mid"]))
		self.add_to_dict("\'", "Splyce", "".join([punct_dir, "musicode_splyce_singlequotationmark.mid"]))
		self.add_to_dict("?", "Splyce", "".join([punct_dir, "musicode_splyce_questionmark.mid"]))
		print("Dictionary(s) established.")


	def _setup_boX_dict(self): # punctuation_only=False):
		self._setup_letters_and_numbers("Baud-Onkadonk_X") #if punctuation_only is False else None
		punct_dir = self.musicode_path + "\\Baud-Onkadonk_X\\boX_Punctuation\\"
		self.add_to_dict("/", "Baud-Onkadonk_X", "".join([punct_dir, "musicode_boX_forwardslash.mid"]))
		#self.add_to_dict("]", "Baud-Onkadonk_X", "".join([punct_dir, "musicode_boX_closebracket.mid"]))
		self.add_to_dict(")", "Baud-Onkadonk_X", "".join([punct_dir, "musicode_boX_closeparenthesis.mid"]))
		self.add_to_dict(":", "Baud-Onkadonk_X", "".join([punct_dir, "musicode_boX_colon.mid"]))
		self.add_to_dict(",", "Baud-Onkadonk_X", "".join([punct_dir, "musicode_boX_comma.mid"]))
		self.add_to_dict('\"', "Baud-Onkadonk_X", "".join([punct_dir, "musicode_boX_doublequotationmark.mid"]))
		self.add_to_dict("!", "Baud-Onkadonk_X", "".join([punct_dir, "musicode_boX_exclamationmark.mid"]))
		self.add_to_dict("-", "Baud-Onkadonk_X", "".join([punct_dir, "musicode_boX_hyphen.mid"]))
		#self.add_to_dict("[", "Baud-Onkadonk_X", "".join([punct_dir, "musicode_boX_openbracket.mid"]))
		self.add_to_dict("(", "Baud-Onkadonk_X", "".join([punct_dir, "musicode_boX_openparenthesis.mid"]))
		self.add_to_dict(".", "Baud-Onkadonk_X", "".join([punct_dir, "musicode_boX_period.mid"]))
		self.add_to_dict(";", "Baud-Onkadonk_X", "".join([punct_dir, "musicode_boX_semicolon.mid"]))
		self.add_to_dict("\'", "Baud-Onkadonk_X", "".join([punct_dir, "musicode_boX_singlequotationmark.mid"]))
		self.add_to_dict("?", "Baud-Onkadonk_X", "".join([punct_dir, "musicode_boX_questionmark.mid"]))
		print("Dictionary(s) established.")


	def _setup_boY_dict(self): # punctuation_only=False):
		self._setup_letters_and_numbers("Baud-Onkadonk_Y") #if punctuation_only is False else None
		punct_dir = self.musicode_path + "\\Baud-Onkadonk_Y\\boY_Punctuation\\"
		self.add_to_dict("/", "Baud-Onkadonk_Y", "".join([punct_dir, "musicode_boY_forwardslash.mid"]))
		#self.add_to_dict("]", "Baud-Onkadonk_Y", "".join([punct_dir, "musicode_boY_closebracket.mid"]))
		self.add_to_dict(")", "Baud-Onkadonk_Y", "".join([punct_dir, "musicode_boY_closeparenthesis.mid"]))
		self.add_to_dict(":", "Baud-Onkadonk_Y", "".join([punct_dir, "musicode_boY_colon.mid"]))
		self.add_to_dict(",", "Baud-Onkadonk_Y", "".join([punct_dir, "musicode_boY_comma.mid"]))
		self.add_to_dict('\"', "Baud-Onkadonk_Y", "".join([punct_dir, "musicode_boY_doublequotationmark.mid"]))
		self.add_to_dict("!", "Baud-Onkadonk_Y", "".join([punct_dir, "musicode_boY_exclamationmark.mid"]))
		self.add_to_dict("-", "Baud-Onkadonk_Y", "".join([punct_dir, "musicode_boY_hyphen.mid"]))
		#self.add_to_dict("[", "Baud-Onkadonk_Y", "".join([punct_dir, "musicode_boY_openbracket.mid"]))
		self.add_to_dict("(", "Baud-Onkadonk_Y", "".join([punct_dir, "musicode_boY_openparenthesis.mid"]))
		self.add_to_dict(".", "Baud-Onkadonk_Y", "".join([punct_dir, "musicode_boY_period.mid"]))
		self.add_to_dict(";", "Baud-Onkadonk_Y", "".join([punct_dir, "musicode_boY_semicolon.mid"]))
		self.add_to_dict("\'", "Baud-Onkadonk_Y", "".join([punct_dir, "musicode_boY_singlequotationmark.mid"]))
		self.add_to_dict("?", "Baud-Onkadonk_Y", "".join([punct_dir, "musicode_boY_questionmark.mid"]))
		print("Dictionary(s) established.")


	# MUSICODE_FUNCTIONS
	# --------------------------------------
	# -----------------------------------------------------------------------
	# ----------------------------------------------------------------------------------------------------------------------

	#UA-1.
	def translate(self, musicode, string):
		"""

			Translates a string into a selected musicode.

		:param musicode: 	The musicode to translate into
		:param string: 		The string text to translate
		:return: 			music21 stream containing the translated text.
		"""
		if musicode != self.musicode_name:		#When you use "User_Selected" as a musicode choice,
			s = music21.stream.Part()			#you can only use the measures you've created using make_musicode.
			num = 0
			for c in string:
				new_measure = self._translate_letter(c, musicode, num)
				s.append(new_measure)
				num = num + 1
			return s
		elif musicode == self.musicode_name:
			s = music21.stream.Part()
			num = 0
			for c in string:
				new_measure = self._translate_letter(c, musicode, num)
				print("New_Measure:", new_measure)
				s.append(new_measure)
				num = num + 1
			return s
		#else:
			#for c in string:



	#UA-2.
	def translate_multiline(self, musicode, file_name, write=False,
							write_name = "Multiline_Musicode_Write", from_text_file=False):
		"""
			Translates input from a text file.

		:param musicode: 	The user-selected musicode.
		:param file_name: 	The file name OR the string to translate if from_text_file is False.
		:return: 			music21.stream.Stream full of music21.stream.Part()s for every "\n".
		"""
		# TODO: Find a way to add option to split each line of text into a different midi track
		if from_text_file:
			file = open(file_name, "r")
			string = file.read()

		else:
			string = file_name

		s = music21.stream.Part()
		num = 0
		line = 1
		#print("LINE", line)
		ss = music21.stream.Part()
		w = music21.ElementWrapper("Line: %d" % line)
		ss.insert(w)
		ss.partName = line
		#line += 1
		#string_character = file.read(1)
		#while string_character:
		for c in string:
			#print("C", c)
			if c == "\n":
				s.insert(ss)
				ss = music21.stream.Part()
				line = line + 1
				w = music21.ElementWrapper("Line: %d " % line)
				ss.insert(w)
				#print("LINE2", line)
				ss.partName = line
				#print("LINE3", line)
			else:
				#If a dictionary has been setup
				if False in [len(self.dictionaries[i]) is 0 for i in self.dictionaries.keys()]:
					#Check c contents
					if c != "\n":
						#Via an assert.
						assert c in self.dictionaries[
							musicode].keys(), "String character '%s' not a valid musicode character." % c
				else:
					print("Your musicodes dictionaries have not been established yet.")
				new_measure = self._translate_letter(c, musicode, num)
				ss.append(new_measure)
				num = num + 1
			#string_character = file.read(1)
		# for note in ss.flat.notes:
		#     note.show('txt')
		if (len(ss.flat.notes) != 0):
			s.insert(ss)
		if write:
			for i in s:
				if from_text_file:
					name = os.path.splitext(os.path.basename(file_name))[0]
				else:
					if os.path.isfile(write_name):
						name = os.path.splitext(os.path.basename(write_name))[0]
					else:
						name = write_name
				intermediary_path = os.getcwd() + os.sep + "resources" + os.sep + "intermediary_path" + os.sep
				i.write("mid", intermediary_path + "Ln-%s" % i.partName + "_" + name + ".mid" )
			print("Write successful!")
			return s
		else:
			print("Musicode stream successful!")
			return s


	#UA-3.
	def _translate_letter(self, c, musicode, num = None):
		"""
			Translates a single "letter" string into a selected musicode. Adapted to user-generated musicode from the
		make_musicode() function.

		:param c: 			A single string instance. (i.e  "y" or "!")
		:param musicode: 	The desired Musicode to be called.
		:param num: 		The measure's measureNumber.
		:return:
		"""
		#new_measure = music21.stream.Measure()
		# m.show('txt')
		# print("letter:" + c + "\n")

		if musicode == self.musicode_name:
			#self.dictionaries.update(User_Generated=self.userCreated)
			print("C:", c)
			print("Here")
			print("Here 2:", self.user_Created)
			#self.userCreated.show()
			for measure in self.user_Created:
				print("How bout here?")
				print("User_Created:")
				self.user_Created.show('txt')

				#TODO Write this block differently?
				# element_wrapper = measure[-1]
				# if type(element_wrapper) == music21.note.Rest:
				# 	element_wrapper = measure[-2]
				# elif type(element_wrapper) == music21.bar.Barline:
				# 	element_wrapper = measure[-3]
				# elif type(element_wrapper) == music21.note.Note:
				# 	element_wrapper = measure[-2]
				# else:
				# 	pass

				#This call's list should always have only one element for our purposes.
				# (Unless other elementwrappers are added to these musicode measures, for some unknown reason.)
				element_wrapper = [i for i in measure.getElementsByClass(["ElementWrapper"])][0]

				print("Measure")
				measure.show('txt')
				print("Element_wrapper", element_wrapper)
				print("Element_wrapper_object:", element_wrapper.obj)

				if element_wrapper.obj == c:
					new_measure = copy.deepcopy(measure)
					print("New_Measure:", new_measure)
					return new_measure
				elif element_wrapper.obj not in c:
					print("Your user-generated musicode does not have this '%s' string character:" % c)


		elif musicode != self.musicode_name:
			#self.dictionaries.update(User_Generated=self.userCreated)

			rt = self.dictionaries[musicode].get(c)
			#print("RT1", rt)
			eval(self.setup_pairs[musicode])() if rt is None else None
			#print("rt", rt)
				#self._setup_letters_and_numbers(musicode)
			rt = self.dictionaries[musicode].get(c)
			#print("rt1.5", rt)
			#assert rt is not None, "rt is still none.....bug."
			if rt:
				#print("RT2", rt)
				new_measure = copy.deepcopy(rt)
				#print("NEW_MEASURE1", new_measure)
				return new_measure
			else:
				print("Here7")
				try:
					new_measure = copy.deepcopy(self.dictionaries[musicode].get(" "))
					#print("NEW_MEASURE2", new_measure)
					new_measure.number = num
					return new_measure
				except AttributeError as i:
					#print("Attr Error:", i)
					space_measure = music21funcs.empty_measure()   #NOTE: timesig won't always be set here. Be aware.
					self.dictionaries[musicode].update([(" ", space_measure)])
					new_measure = copy.deepcopy(self.dictionaries[musicode].get(" "))
					new_measure.number = num
					return new_measure

	#UA-4.
	def translate_each_letter_to_random_musicode(self, text):
		"""
			Translates the text into a stream where each letter is a random Musicode.

		:param text: 	The text to be translated.
		:return: 		Stream.
		"""
		s = music21.stream.Part()
		num = 0
		musicodes = list(self.dictionaries.keys())
		for c in text:
			musicode = musicodes[random.randint(0, len(musicodes) - 1)]
			new_measure = self._translate_letter(c, musicode, num)
			new_measure.number = num
			s.append(new_measure)
			num = num + 1
		return s

	#UA-5.
	def align_musicode_with_melody(self, melody_stream, musicode_stream, keysig=None):
		"""

			This function takes a composed melody and intertwines it with a translated instance of musicode. See picture
		#TODO figure out pictures with doc strings. In the alignment process, spaces are ignored.
		The number of melody_stream.flat.notes must match the number of musicode_stream.getElementsByClass(-
		--stream.Measure).

		:param melody_stream: 	The selected melody for alignment. Generally acquired from a composed instance of midi
								converted with converter.parse.
		:param musicode_stream: The text to be aligned with the melody. Acquired from musicode.mc.translate("Musicode",
								"String")
		:return: 				New music21.stream.Stream object of aligned melody stream.
		#TODO The keyAware transpostion of the letters in the measures needs work with respect to bass and root of chords.
		"""

		if type(melody_stream) == music21.stream.Score:
			melody_stream = melody_stream.pop(0)    #Converter.parse returns a score with parts. We want the stream.Part which has our just-composed melody.
		keysig = music21.key.Key(keysig)		#If keysig == None, key.Key will default to C.
		melody_stream.insertAndShift(0.0, keysig)     #Inserts key.Key at begining of part stream.
		s = music21.stream.Stream()
		m = 0
		musicode_measures = musicode_stream.getElementsByClass(music21.stream.Measure)
		for n in melody_stream.flat.notes:
			if m < len(musicode_measures):
				ms = music21.stream.Stream()
				copied_measure = copy.deepcopy(musicode_measures[m])
				# 1. Set musicode offset to match note offset.

				# 2. Set musicode duration to match note duration. Use stretch function.
				temp = music21.stream.Stream()
				for y in musicode_measures[m].flat.notes:
					temp.insert(y)
				m_dur = temp.highestTime
				n_dur = float(n.quarterLength)

				ms.insert(0, copied_measure)
				copied_measure.number = 0

				ms.show('txt')
				new_measure = music21funcs.stretch_by_measure(ms, 0, 0, (n_dur / (m_dur)))
				# print("cunt")
				new_measure.show('txt')
				# 3. Set musicode root notes pitch to match note pitch. Use transpose key aware.
				# A. Get interval between musicode root note pitch and current melody note's pitch.
				if type(new_measure.flat.notes[0]) is music21.chord.Chord:
					p1 = new_measure.flat.notes[0].root()
				else:
					p1 = new_measure.flat.notes[0]
				if type(n) is music21.chord.Chord:
					print("Melody must be notes; it cannot be chords.")
					return None
				else:
					p2 = n
				i = music21.interval.notesToGeneric(p1, p2)
				# B. GET KEY.
				if len(melody_stream.getKeySignatures()) != 1:
					print("You have no key signature, or more than one.")
					return None
				else:
					k = melody_stream.getKeySignatures()[0].asKey()
				# C. Transpose Key aware all of new measure by THAT interval.
				for z in new_measure.flat.notes:
					if type(z) is music21.note.Note:
						i.transposePitchKeyAware(z.pitch, k, inPlace=True)
					elif type(z) is music21.chord.Chord:
						for p in z.pitches:
							i.transposePitchKeyAware(p, k, inPlace=True)
					else:
						print("You messed up. Type of z is wrong.")
						print(type(z))
					new_measure.show('txt')
				# 4. Insert into stream.
				s.insert(n.offset, new_measure)

			m = m + 1

		s.makeMeasures()
		return s

	def filter_to_NOT_gray_or_white(self, img):
		indices = np.where(img != [255, 255, 255])
		coordinates = zip(indices[0], indices[1])
		coordinate_list = [i for i in OrderedDict.fromkeys(coordinates).keys()] #OrderedDict gets rid of duplicates.
		# For every coordinate that is not a white color tuple....
		for i in coordinate_list:
			if list(img[i[0], i[1]]) == [204, 204, 204] or list(img[i[0], i[1]]) == [255, 255, 255]:
				pass
			else:
				img[i[0], i[1]] = np.array([255, 255, 255])


	def get_cropped_empty_box_ROI_compensation(self):
		###Compensation charactar box
		compensation_bounding_box = cv2.imread(r".\resources\empty_box.png")
		operation_image = cv2.imread(r".\resources\empty_box_compensation.png")
		#self.filter_to_NOT_gray_or_white(compensation_bounding_box)
		compensation_gray = cv2.cvtColor(operation_image, cv2.COLOR_BGR2GRAY)
		comp_thresh, comp_bnwi = cv2.threshold(compensation_gray, 127, 255, cv2.THRESH_BINARY)
		image = compensation_bounding_box
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
		_a, cntrs, heirarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		x, y, w, h = cv2.boundingRect(cntrs[0])
		#cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
		ROI = comp_bnwi[y:y + h, x:x + w]

		#cv2.imshow('ROI_before_crop', ROI)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()

		th, threshed = cv2.threshold(ROI, 240, 255, cv2.THRESH_BINARY_INV)
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
		morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
		cnts2 = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
		cnt = sorted(cnts2, key=cv2.contourArea)[-1]
		x2, y2, w2, h2 = cv2.boundingRect(cnt)
		cropped_ROI = ROI[y2:y2 + h2, x2:x2 + w2]
		#cv2.imshow('ROI_AFTER_crop', cropped_ROI)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
		return cropped_ROI


	####FINAL FUNCTION
	def create_musicode_unicode_master_dict_and_stream(self, mypath=r".\resources\unicode_rapid_tables\\"):
		#originalImage = cv2.imread(r"C:\Users\Isaac's\Desktop\PicPick AutoSave-T\Image 1584.png")
		#img = originalImage
		start = 0
		unicode_master_dict = OrderedDict()
		unicode_master_stream = music21.stream.Stream()
		files = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

		for i in files:
			master_tuple = self.create_musicodes_from_unicode_rapidtable(cv2.imread(i), start_index=start)
			#unicode_master_dict.update(master_tuple[0])
			#unicode_master_stream.append(master_tuple[1])
			print("Rapid Table of unicode characters %s" % str(start) + "-%s" % str(start + 256) + " processed successfully!")
			start += 256
		return (unicode_master_dict, unicode_master_stream)



	def create_musicodes_from_unicode_rapidtable(self, img, start_index):

		picklepath = os.path.abspath(r"C:\Users\Isaac's\Midas\resources\pickles\unicode_stream_pickles")


		###Our instantiation image data.
		originalimage1 = copy.deepcopy(img)
		#originalimage2 = copy.deepcopy(img)
		###Get a grey image for filtering purposes.
		grayImage = cv2.cvtColor(originalimage1, cv2.COLOR_BGR2GRAY)
		###Get our main black and white image.
		thresh, bnwi = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

		###Get color-filtered rapid table image from MAIN img in order to acquire appropriate bounding box data.
		###Change all color pixels that aren't the grey borders or white background to white here. Happens in place.
		self.filter_to_NOT_gray_or_white(img)

		#For Testing#
		# cv2.imshow('Chessboard, white with grey borders.', img)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		#TODO copy.deepcopy here?
		image = img
		# original = image.copy() ##this be a copy of original filtered grid
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  #Critical.

		###Acquire our contours (cnts) which allows us to acquire our bounding box data for our Regions Of Interest.
		_a, cnts, heirarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		# cnts = cnts[0] if len(cnts) == 2 else cnts[1]

		###Get rid of the "global" bounding box, the border of the image surrounding all the little boxes.
		cnts.pop(-1)

		###Counter variables.
		###Rapid tables are 16X16 tables of unicode character points. Every time one is parsed, we up the counter by 256
		end_index = start_index + 256
		inverse_index = end_index - 1
		ROI_number = 0

		###Our OrderedDict that will be returned at the end. Established with a specific range of unicode values.
		unicode_dict = OrderedDict.fromkeys([chr(i) for i in range(start_index, end_index, 1)])
		unicode_stream = music21.stream.Part()   #All 256 rapid tables will be their own music21.stream.Part()

		###Core
		print("Core")
		assert len([i for i in range(start_index, end_index, 1)]) == 256, "Len of values range should be 256."
		print("Core start.")
		for i in range(start_index, end_index, 1):    # in cnts:
			try:
				print("I -->", i)
				###Acquire our REGION OF INTEREST from our filtered rapid table.
				x, y, w, h = cv2.boundingRect(cnts[i - start_index])      ### if i > 256 else cv2.boundingRect(cnts[i])]
				cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)  # Not needed? Draws that green line.
				ROI = bnwi[y:y + h, x:x + w]
				#cv2.imwrite(r"C:\Users\Isaac's\Desktop\ROI_Folder\\" + 'ROI_{}.png'.format(ROI_number), ROI)

				#####--->Filter out whitespace around unicode character in our ROI.
				### (1) Convert to gray, and threshold
				#gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)  # img
				th, threshed = cv2.threshold(ROI, 240, 255, cv2.THRESH_BINARY_INV)
				### (2) Morph-op to remove noise
				kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
				morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
				### (3) Find the max-area contour
				cnts2 = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
				#try:
				cnt = sorted(cnts2, key=cv2.contourArea)[-1]
				#except IndexError as i:
					#print("Index Error", i)

					#return unicode_dict
				### (4) Crop our new ROI!! (This is our unicode character WITHOUT the surrounding white color tuples)
				x2, y2, w2, h2 = cv2.boundingRect(cnt)
				cropped_ROI = ROI[y2:y2 + h2, x2:x2 + w2]
			except IndexError as i:
				print("Index Error", i)
				print("Unicode character %s on rapidtable is found to be empty. "
					  "Compensating with 'nac - empty box' as character." % chr(inverse_index))
				cropped_ROI = self.get_cropped_empty_box_ROI_compensation()

			#cv2.imwrite(r"C:\Users\Isaac's\Desktop\ROI_Folder\\" + 'cropped_ROI_{}.png'.format(ROI_number), cropped_ROI)

			#TODO conversion necessary?
			#conversion = cv2_tuple_reconversion(cropped_ROI,
			#									inPlace = False,
			#									conversion = 'Monochrome')
			#preview = conversion[1]
			#preview2 = conversion[0]

			#####Midiart transformation -- monochrome method turning black pixels into midi notes.
			###Get music21.stream
			stream = make_midi_from_grayscale_pixels(cropped_ROI,
													 granularity = .125,  # 32nd note
													 connect = False,  # dialog.chbxConnect.GetValue(), ###MIDAS_Settings.connectNotes,
													 note_pxl_value = 0)
			stream.makeMeasures(inPlace=True)
			#print("Here 4")
			assert stream.hasMeasures(), "Your stream does not have measures. Check for problems."
			###Get the measure with our musical data.
			measure = stream[0]
			#print("MEASURE", measure)
			###Create an element wrapper with the corresponding musicode unicode str in it.
			ele = music21.ElementWrapper(obj=chr(inverse_index))
			measure.append(copy.deepcopy(ele))

			###Cap the measure with correct rests.
			music21funcs.fill_measure_end_gaps(measure, inPlace=True)  # TODO Time consideration in future.

			# WE DO INVERSE INDEX method BECAUSE Our ROIs are acquired from bottom right of our image(s) to top left.
			###Update dict where key==unicode "chr" and value == musicode stream.measure()

			#print("Here4.5")
			unicode_dict[chr(inverse_index)] = measure
			measure.number = inverse_index
			#print("MEASURE NUMBER HERE-->", measure.number)
			unicode_stream.insert(inverse_index*4, measure)
			#print("Here4.7")

			inverse_index -= 1
			# stream.write('mid', )
			#print("Here5")
			ROI_number += 1
		#print(os.path.dirname(r".\resources"))
		music21.converter.freeze(unicode_stream,
								 fp = picklepath + os.sep +
									  "unicode_musicode_pickle---CHARACTERS(%s" % str(start_index) + "-%s).p" % str(end_index-1))
		#unicode_stream.makeMeasures(inPlace=True) #Purpose of this is to sort the already-made measures in the stream.
		return (unicode_dict, unicode_stream)


	def create_directories(self):
		if os.path.exists(self.full_new_musicode_path) is False:
			os.mkdir(self.full_new_musicode_path)
		os.mkdir(self.full_new_musicode_path + "\\" + self.sh + "_" + "Uppercase\\")
		os.mkdir(self.full_new_musicode_path + "\\" + self.sh + "_" + "Lowercase\\")
		os.mkdir(self.full_new_musicode_path + "\\" + self.sh + "_" + "Numbers\\")
		os.mkdir(self.full_new_musicode_path + "\\" + self.sh + "_" + "Punctuation\\")

	# M21-7.
	def make_musicode(self, in_stream, musicode_name="User_Generated", shorthand="sh", filepath=None,
							selection=None, write=True, timeSig='4/4'):
		"""
			This function takes a string and an input stream with equal lengths (# of measures == # of string characters)
		and zips them together to create a user-generated assignment of 'musicodes' characters that can then be called
		by its respective string character.
		Choice included to write to file and to provide a selection. Selection should ideally match input stream,
		although creativity is highly encouraged!

		:param in_stream: 		Operand music21.stream.Stream() object with musicode data IN MEASURES to be written to
								file.
		:param musicode_name: 	The name of your user-created and designed 'musicode' to be generated from said stream
								of measures.
		:param shorthand: 		The abbreviation for your musicode. (i.e, The builtin musicode "Animuse" uses the
								shorthand 'am'.)
		:param filepath: 		If this is none, the musicode will be saved in the ...
								\\Midas\\resources\\musicode_libraries folder.
								#NOTE: Not recommended to name musicode the same existing....
		:param selection: 		If none, select is automatically the Latin_Script established as--
		#--Latin_Script = '''AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz  ?,;\':-.!\"()[]/  0123456789'''
		:param write: 			If true, directories are established appropriately and function writes to them
								accordingly.
		:param timeSig: 		Determines the time signature for the measures of your new musicode. Primarily affects
								measure length.
		:return: 				Returns a stream containing your musicode with its assigned element wrappers for quick
								access.
		"""

		#The name of the created musicode.
		self.musicode_name = musicode_name
		#The created musicodes abbreviation. (i.e "Animuse" and "am")
		self.sh = shorthand

		# User string 'selection' condition check.
		#Critical: Selection == in_stream Length Check:
		if selection is not None:   		##NOTE: If your selection is None, this code assumes that you are
			selection = selection			#creating a musicode FOR EVERY character in the Latin_Script.
		elif selection is None:
			selection = self.Latin_Script
		print("User Selection:", selection)


		#Assert for measures.
		in_stream.makeMeasures(inPlace=True)  #TO BE BLESSED SURE.
		assert in_stream.hasMeasures(), "Make measures inPlace failed."
		if in_stream.hasMeasures() is False:
			in_stream = in_stream.makeMeasures(inPlace=False)
		assert in_stream.hasMeasures(), "There are no measures in this stream. Call 'in_stream.makeMeasures().'"
		assert in_stream[0].isMeasure, "This first index is not a music21.stream.Measure() object."

		#Assert for equal zip lengths.
		if len(selection) != len(in_stream):
			print("Selection", len(selection))
			print("In_Stream", len(in_stream))
			print("Your selected string assignment does not match the length of your created musicode.")
			print("Musicode creation failed.")
			return None

		# Establish Writing Path Name (created on call of function whether writing or not, overwritten if already existing)
		#if write is not False:
		set_path = r"musicode_libraries"
		if filepath is None and write is True:  #Then we write to the r".\Midas\resources\musicode_libraries\ folder.
			filepath = set_path
			absFilePath = os.path.dirname(os.path.abspath(set_path))
			resource_path = absFilePath + "\\resources\\" + filepath + "\\"
			if os.path.exists(resource_path + musicode_name + "\\\\") is True:  #Catch for fileexists error. Remove and rewrite, since will be commonly called.
				shutil.rmtree(resource_path + musicode_name + "\\\\", ignore_errors=True)  #Caution with this?
			os.mkdir(resource_path + os.sep + musicode_name + "\\")
			self.full_new_musicode_path = resource_path + "\\" + musicode_name + "\\"
			print("Full_New_Musicode_Path created.", self.full_new_musicode_path)
			self.create_directories()		#* Create Directories within Path since write and file_path are true, and we're writing to it.
		# print(resource_path + musicode_name + "\\")

		elif filepath is not None and write is True:  ##This block executes to save to specified fullpath.
			#example: r"C:\Users\User\Midas\resources\musicode_libraries    NOTE: No extra slashes.
			self.full_new_musicode_path = filepath + os.sep + self.musicode_name

			print("FNMP", self.full_new_musicode_path)
			self.create_directories()        #Create directories...*
		else:
			pass


		#Length Checks, once makeMeasures has taken place.
		print("In_Stream Length", len(in_stream))
		print("Selection Length:", len(selection))


		# Make new stream with measures derived from selection:
		self.new_stream = music21.stream.Stream()

		# Make a TimeSignature object. Does not get appended to a stream, but it is important.
		time = music21.meter.TimeSignature(timeSig)

		#Core operation.
		zipped_selection_and_measures = zip(selection, in_stream)
		self.zip_dict = OrderedDict([i for i in zipped_selection_and_measures])
		for i in self.zip_dict.keys():
			self.zip_dict[i].append(music21.ElementWrapper(obj=i))  #Note: measure.append calls return none, so they  must happen inPlace.
			#Handle incomplete measure\gaps issue with added rests and-
			#-reset measures in new_stream, rewriting its measure numbers in the process, just in case.
			music21funcs.fill_measure_end_gaps(self.zip_dict[i], time, inPlace=True)
			self.new_stream.append(self.zip_dict[i])

		#Checks
		print("NEW_STREAM:")
		self.new_stream.show('txt')
		print("NEW_STREAM_LENGTH", len(self.new_stream))

		#If write==True,  Write to established directories:
		#These exports are based on the old method of importing from individual midi files.
		#TODO Export new_stream method.
		try:
			#TODO rewrite using getElementByClass method from above _translate method. 02/11/
			for j in range(0, len(self.new_stream)):
				# new_stream[j].append(wrapper_list[j])
				element_wrapper = self.new_stream[j][-1]  # The last element in each measure.
				if type(element_wrapper) == music21.note.Rest:
					element_wrapper = self.new_stream[j][-2]
				if type(element_wrapper) == music21.bar.Barline:
					element_wrapper = self.new_stream[j][-3]
				# print(new_stream[j].measureNumber, stringz.obj)
				#print("X", [self.new_stream[j]])
				# A check against writing empty measures and whether to write at all.
				#print("ELEMENT WRAPPER,", element_wrapper)
				if element_wrapper.obj is not ' ' and write is not False:
					if self.new_stream[j].hasElementOfClass(music21.note.Note) or self.new_stream[j].hasElementOfClass(
							music21.chord.Chord):
						if element_wrapper.obj.islower() and element_wrapper.obj not in self.Punct_Names:
							self.new_stream[j].write("mid",
												self.full_new_musicode_path + "\\" + self.sh + "_" + "Lowercase\\" + "musicode" + "_" + self.sh + "_" + str(
													element_wrapper.obj) + ".mid")
						elif element_wrapper.obj in self.Punct_Names or element_wrapper.obj in self.Punct_Symbols:
							self.new_stream[j].write("mid",
												self.full_new_musicode_path + "\\" + self.sh + "_" + "Punctuation\\" + "musicode" + "_" + self.sh + "_" + str(
													self.Punct_Workaround[element_wrapper.obj]) + ".mid")
						elif element_wrapper.obj.isupper():
							self.new_stream[j].write("mid",
												self.full_new_musicode_path + "\\" + self.sh + "_" + "Uppercase\\" + "musicode" + "_" + self.sh + "_" + str(
													element_wrapper.obj) + ".mid")
						elif element_wrapper.obj.isdigit():
							self.new_stream[j].write("mid",
												self.full_new_musicode_path + "\\" + self.sh + "_" + "Numbers\\" + "musicode" + "_" + self.sh + "_" + str(
													element_wrapper.obj) + ".mid")
					else:
						pass

		except AttributeError as i:
			print("i")
			print("Your stream may not have measures correctly. \n"
				  "Call makeMeasures() --> input_stream.makeMeasures(inPlace=True)")
			pass


		#This call is for "translate musicode" only, which writes to directories. It is not for "create musicode."
		if filepath is None and write is False:
			print("Creating and storing new musicode measures....")
			pass
		else:
			print("Midi writing...", self.full_new_musicode_path + os.sep + self.musicode_name + ".mid")
			self.new_stream.write('mid', self.full_new_musicode_path + os.sep + self.musicode_name + ".mid")

		# Account for the " " space string manually, so user is encouraged not to set a musicode for a space.
		# Append "space_measure" to user-generated musicode.
		space_measure = music21funcs.empty_measure(timeSig)
		self.new_stream.append(space_measure)  #TODO IF THEY ADD MANY SPACES, WE"LL NEED TO ACCOUNT FOR THEM. OrderedDict filters all out but one, so if we add another here......

		print("FINAL_NEW_STREAM:")
		#self.new_stream.show('txt')
		self.user_Created = self.new_stream
		print("Musicode Created.")
		return self.zip_dict, self.new_stream
