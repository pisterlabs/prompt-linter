#!/usr/bin/python3

# ## {R2D2919B742E} ##
# ###########################################################################
# What if magic existed?
# What if a place existed where your every thought and dream come to life.
# There is only one catch: it has to be written down.
# Such a place exists, it is called programming.
#    - Scott Taylor Reph, RightThumb.com
# ###########################################################################
# ## {C3P0D40fAe8B} ##


##################################################
import sys, time
##################################################
import _rightThumb._construct as __
appDBA=__.clearFocus(__name__,__file__);__.appReg=appDBA;
def focus(parentApp='',childApp='',reg=True):
	global appDBA;f=__.appName(appDBA,parentApp,childApp);
	if reg:__.appReg=f;
	return f
import _rightThumb._base3 as _
fieldSet=_.l.vars(focus(),__name__,__file__,appDBA)
_.load()
##################################################
_v = __.imp('_rightThumb._vars')
_str = __.imp('_rightThumb._string')
##################################################


def sw():
	pass
	#b)--> examples
	# _.switches.register( 'Input', '-i' )
	# _.switches.register( 'URL', '-u,-url,-urls', 'https://efm.cx/', isData='raw' )
	#e)--> examples
	# _.switches.register( 'Files', '-f,-fi,-file,-files','file.txt', isData='name,data,clean', description='Files', isRequired=False )

# __.setting('require-list',['Files,Plus','File,Has']) # todo
# __.setting('require-list',['Pipe','Files'])
__.setting('receipt-log')
__.setting('receipt-file')
__.setting('myFileLocations-skip-validation',False)
__.setting('require-pipe',False)
__.setting('require-pipe||file',False)
__.setting('pre-error',False)
__.setting('switch-raw',[])



_.appInfo[focus()] = {
	# 'app': '8facG-jo0Cxk',
	'file': 'thisApp.py',
	'liveAppName': __.thisApp( __file__ ),
	'description': 'Changes the world',
		# _.ail(1,'subject')+
		# _.aib('one')+
	'categories': [
						'DEFAULT',
				],
	'usage': [
						# 'epy another',
						# 'e nmap',
						# '',
	],
	'relatedapps': [
						# 'p another -file file.txt',
						# '',
	],
	'prerequisite': [
						# 'p another -file file.txt',
						# '',
	],
	'examples': [
						_.hp('p thisApp -file file.txt'),
						_.linePrint(label='simple',p=0),
						'',
	],
	'columns': [
					# { 'name': 'name', 'abbreviation': 'n' },
					# { 'name': '{1}', 'abbreviation': '{0}', 'sort': '{2}' },
	],
	'aliases': [
					# 'this',
					# 'app',
	],
	'notes': [
					# {},
	],
}

_.appData[focus()] = {
		'start': __.startTime,
		'uuid': '',
		'audit': [],
		'pipe': False,
		'data': {
					'field': {'sent': [], 'received': [] }, # { 'label': '', 'context': [],  }
					'table': {'sent': [], 'received': [] },
		},
	}


def triggers():
	_.switches.trigger( 'Files', _.myFileLocations, vs=True )
	_.switches.trigger( 'Ago', _.timeAgo )
	_.switches.trigger( 'Folder', _.myFolderLocations )
	_.switches.trigger( 'URL', _.urlTrigger )
	_.switches.trigger( 'Duration', _.timeFuture )

def _local_(do): exec(do)

_.l.conf('clean-pipe',True)
_.l.sw.register( triggers, sw )

########################################################################################
#b)--> examples
#d)--> code hints to quickly get started
	#n)--> inline examples
		# any(ele in 'scott5' for ele in list('0123456789'))
		# if _.switches.isActive('Test'): test(); return None;
		# result=[]; result=[ _.pr(line) for i, line, bi in _.numerate( _.isData(r=0) )]
		# bk=[];[  bk.append(rec['backup']) for rec in backupLog if path == rec['file']]; bk=bk[-1];
		# a=(1 if True else 0) <--# 
		#!)--> m=[[row[i] for row in matrix] for i in range(4)]

	#n)--> python globals
		# globals()['var']
		# for k in globals(): print(k, eval(k) )

	#n)--> webpage from url
		# for subject in _.caseUnspecific( line, needle ): line = line.replace( subject, _.colorThis( subject, 'green', p=0 ) )

	#n)--> webpage from url
		# requests=__.imp('requests.post')
		#!)--> data=str(requests.post(url,data={}).content,'iso-8859-1')

	#n)--> import and backup example
		# _bk = _.regImp( __.appReg, 'fileBackup' ); _bk.switch( 'Silent' ); _bk.switch( 'isRunOnce' ); _bk.switch( 'Flag', 'APP' ); _bk.switch( 'DoNotSchedule' )
		# _bk.switch( 'Input', path ); bkfi = _bk.action();
	
	#n)--> inline
		# for rel in [ subject for subject in _.isData(r=0) if _.showLine(subject) ]: print(rel)

	#n)--> banner
		# banner=_.Banner(app); goss=banner.goss;
#e)--> examples
########################################################################################
#n)--> start

def action():


	import os
	import openai
	# import json

	openai.organization = os.getenv("OPENAI_API_ORG")
	openai.api_key = os.getenv("OPENAI_API_KEY")

	response = openai.Model.list()
	models = response['data']

	for model in models:
		if 'root' in model:
			print(model.root)


"""
	babbage
	davinci
	text-davinci-edit-001
	babbage-code-search-code
	text-similarity-babbage-001
	code-davinci-edit-001
	text-davinci-001
	ada
	babbage-code-search-text
	babbage-similarity
	whisper-1
	code-search-babbage-text-001
	text-curie-001
	code-search-babbage-code-001
	text-ada-001
	text-embedding-ada-002
	text-similarity-ada-001
	curie-instruct-beta
	ada-code-search-code
	ada-similarity
	text-davinci-003
	code-search-ada-text-001
	text-search-ada-query-001
	davinci-search-document
	ada-code-search-text
	text-search-ada-doc-001
	davinci-instruct-beta
	text-similarity-curie-001
	code-search-ada-code-001
	ada-search-query
	text-search-davinci-query-001
	curie-search-query
	gpt-3.5-turbo-0301
	davinci-search-query
	babbage-search-document
	ada-search-document
	text-search-curie-query-001
	text-search-babbage-doc-001
	gpt-3.5-turbo
	curie-search-document
	text-search-curie-doc-001
	babbage-search-query
	text-babbage-001
	text-search-davinci-doc-001
	text-search-babbage-query-001
	curie-similarity
	curie
	text-similarity-davinci-001
	text-davinci-002
	davinci-similarity
	cushman:2020-05-03
	ada:2020-05-03
	babbage:2020-05-03
	curie:2020-05-03
	davinci:2020-05-03
	if-davinci-v2
	if-curie-v2
	if-davinci:3.0.0
	davinci-if:3.0.0
	davinci-instruct-beta:2.0.0
	text-ada:001
	text-davinci:001
	text-curie:001
	text-babbage:001
"""

# sorted
"""
	ada
	ada-code-search-code
	ada-code-search-text
	ada-search-document
	ada-search-query
	ada-similarity
	ada:2020-05-03
	babbage
	babbage-code-search-code
	babbage-code-search-text
	babbage-search-document
	babbage-search-query
	babbage-similarity
	babbage:2020-05-03
	code-davinci-edit-001
	code-search-ada-code-001
	code-search-ada-text-001
	code-search-babbage-code-001
	code-search-babbage-text-001
	curie
	curie-instruct-beta
	curie-search-document
	curie-search-query
	curie-similarity
	curie:2020-05-03
	cushman:2020-05-03
	davinci
	davinci-if:3.0.0
	davinci-instruct-beta
	davinci-instruct-beta:2.0.0
	davinci-search-document
	davinci-search-query
	davinci-similarity
	davinci:2020-05-03
	gpt-3.5-turbo
	gpt-3.5-turbo-0301
	if-curie-v2
	if-davinci-v2
	if-davinci:3.0.0
	text-ada-001
	text-ada:001
	text-babbage-001
	text-babbage:001
	text-curie-001
	text-curie:001
	text-davinci-001
	text-davinci-002
	text-davinci-003
	text-davinci-edit-001
	text-davinci:001
	text-embedding-ada-002
	text-search-ada-doc-001
	text-search-ada-query-001
	text-search-babbage-doc-001
	text-search-babbage-query-001
	text-search-curie-doc-001
	text-search-curie-query-001
	text-search-davinci-doc-001
	text-search-davinci-query-001
	text-similarity-ada-001
	text-similarity-babbage-001
	text-similarity-curie-001
	text-similarity-davinci-001
	whisper-1
"""

##################################################
#b)--> examples
# banner=_.Banner(dependencies)
# goss=banner.goss
# goss('-\t this app will sherlock tf out of any python app or python module')
#e)--> examples
##################################################

########################################################################################
if __name__ == '__main__':
	#b)--> examples

	# banner.pr()
	# if len(_.switches.all())==0: banner.gossip()
	
	#e)--> examples
	action()
	_.isExit(__file__)

