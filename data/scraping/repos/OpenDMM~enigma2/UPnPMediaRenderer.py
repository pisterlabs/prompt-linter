# -*- coding: UTF-8 -*-
# Originally based on http://coherence.beebits.net/browser/trunk/Coherence/coherence/backends/gstreamer_renderer.py
# Which has been licensed under the MIT license.
# Sublicensed under the enigma2 license.
from enigma import iPlayableService, eTimer, eDVBVolumecontrol, eServiceReference
from Components.ServiceEventTracker import ServiceEventTracker
from Components.ResourceManager import resourcemanager
from GlobalActions import globalActionMap
from Tools.HardwareInfo import HardwareInfo
from Tools.Log import Log

from sets import Set

from twisted.internet import defer
from twisted.python import failure

from coherence.upnp.core.soap_service import errorCode
from coherence.upnp.core import DIDLLite
from coherence import log

import coherence.extern.louie as louie
from coherence.extern.simple_plugin import Plugin

from UPnPCore import Statics, Item, removeUPnPDevice

import os

class UPnPPlayer(object):
	def __init__(self, session, handlePlayback=False):
		global globalActionMap #fixme #hack
		self.actionmap = globalActionMap
		self.session = session

		self._handlePlayback = handlePlayback

		self.uri = None
		self.metadata = {}
		self.mimetype = None
		self._state = UPnPMediaRenderer.STATE_IDLE

		self.onStateChange = []
		self.onClose = [] #hack

		self.__poll_pos_timer = eTimer()
		self.__poll_pos_timer_conn = self.__poll_pos_timer.timeout.connect(self.updatePosition)

		self.volctrl = eDVBVolumecontrol.getInstance() # this is not nice

		if self._handlePlayback:
			self.__event_tracker = ServiceEventTracker(screen=self, eventmap={
				iPlayableService.evEOF: self.__onEOF,
			})

	def _stateChanged(self, state, message=None):
		self._state = state
		for fnc in self.onStateChange:
			fnc(message)

	def __onEOF(self):
		self.stopPolling()
		self._stateChanged(UPnPMediaRenderer.STATE_IDLE, UPnPMediaRenderer.MESSAGE_EOF)

	def __onPlay(self):
		self.startPolling()
		self._stateChanged(UPnPMediaRenderer.STATE_PLAYING)

	def __onPause(self):
		self._stateChanged(UPnPMediaRenderer.STATE_PAUSED)

	def __onStop(self):
		self.stopPolling()
		self._stateChanged(UPnPMediaRenderer.STATE_IDLE)

	def getSeekable(self):
		s = self.session.nav.getCurrentService()
		if s:
			seek = s.seek()
			if seek is None or not seek.isCurrentlySeekable():
				return None
			else:
				return seek

		return None

	def getPausable(self):
		service = self.session.nav.getCurrentService()
		if service:
			return service.pause()
		return None

	def load(self, uri, metadata, mimetype=None):
		Log.i("uri=%s\nmimetype=%s\nmetadata=%s" %(uri, mimetype, metadata))
		self.uri = uri
		self.metadata = metadata
		self.mimetype = mimetype

	def play(self, avoidPlayback=False):
		Log.i("Will now play %s" %self.uri)
		if self._handlePlayback and not avoidPlayback:
			if self._state == UPnPMediaRenderer.STATE_PAUSED:
				if self.unpause():
					return
			service = eServiceReference(eServiceReference.idGST, 0, self.uri)
			if self.metadata != None:
				title = self.metadata.get(Statics.META_TITLE, None)
				artist = self.metadata.get(Statics.META_ARTIST, None)
				album = self.metadata.get(Statics.META_ALBUM, None)
				if title != None:
					if artist != None:
						if album != None:
							title = "%s - %s - %s" %(artist, album, title)
						else:
							title = "%s - %s" %(artist, title)
					Log.i("Generated title is '%s'" %title)
					service.setName(title)

			self.session.nav.playService(service)
		self.__onPlay()

	def transition(self):
		self._state = UPnPMediaRenderer.STATE_TRANSITIONING
		self.stopPolling()

	def pause(self):
		Log.i()
		if self._handlePlayback:
			pausable = self.getPausable()
			if pausable is not None:
				self.stopPolling()
				pausable.pause()
				return True;
				self.__onPause()
		return False

	def seek(self, position):
		Log.i("position=%s" %position)
		self.stopPolling() #don't send a new position until we're done seeking
		seekable = self.getSeekable()
		if seekable != None:
			seekable.seekTo( long(int(position) * 90000) )

		self.startPolling() #start sending position again

	def stop(self, isEOF):
		if isEOF:
			Log.i("EOF")
			self.__onEOF()
		else:
			Log.i()
			if self._handlePlayback:
				self.session.nav.stopService()
			self.__onStop()

	def unpause(self):
		Log.i()
		if self._handlePlayback:
			pausable = self.getPausable()
			if pausable is not None:
				pausable.unpause()
				self.__onPlay()
				return True
			else:
				return False
		else:
			self.__onPlay()
			return True

	def getMute(self):
		Log.i()
		return self.volctrl.isMuted()

	def mute(self):
		Log.i()
		if not self.volctrl.isMuted():
			self.actionmap.actions["volumeMute"]()

	def unmute(self):
		Log.i()
		if self.volctrl.isMuted():
			self.actionmap.actions["volumeMute"]()

	def getVolume(self):
		Log.i()
		return self.volctrl.getVolume()

	def setVolume(self, vol):
		Log.i()
		self.volctrl.setVolume(vol, vol)

	def getState(self):
		#print "[UPnPPlayer.getState]"
		return self._state

	def startPolling(self):
		Log.i()
		self.__poll_pos_timer.stop()
		self.__poll_pos_timer.start(1000, False)

	def stopPolling(self):
		Log.i()
		self.__poll_pos_timer.stop()

	def updatePosition(self):
		for fnc in self.onStateChange:
			fnc()

	def getPosition(self):
		#print "[UPnPPlayer.getPosition]"
		seek = self.getSeekable()
		if seek is None:
			return 0
		pos = seek.getPlayPosition()
		if pos[0]:
			return 0
		return pos[1]

	def getLength(self):
		#print "[UPnPPlayer.getLength]"
		seek = self.getSeekable()
		if seek is None:
			return 0
		length = seek.getLength()
		if length[0]:
			return 0
		return length[1]

	def getUri(self):
		return self.uri

class UPnPMediaRenderer(log.Loggable, Plugin):
	STATE_TRANSITIONING = "transitioning"
	STATE_IDLE = "idle"
	STATE_PLAYING = "playing"
	STATE_PAUSED = "paused"

	MESSAGE_EOF = "eof"
	MESSAGE_BUFFERING = "buffering"

	logCategory = 'enigma2_player'
	implements = ['MediaRenderer']
	vendor_value_defaults = {'RenderingControl': {'A_ARG_TYPE_Channel':'Master'},
							'AVTransport': {'A_ARG_TYPE_SeekMode':('ABS_TIME', 'REL_TIME', 'TRACK_NR')}}
	vendor_range_defaults = {'RenderingControl': {'Volume': {'maximum':100}}}

	def __init__(self, device, session, **kwargs):
		self.name = kwargs.get('name', '%s' %(HardwareInfo().get_device_name()) )
		self.metadata = None
		self.player = kwargs.get('player', UPnPPlayer(session, handlePlayback=True))
		self.player.onStateChange.append(self.update)
		self.tags = {}
		self.server = device
		self.playcontainer = None
		self.dlna_caps = ['playcontainer-0-1']
		louie.send('Coherence.UPnP.Backend.init_completed', None, backend=self)

	def __repr__(self):
		return str(self.__class__).split('.')[-1]

	def update(self, message=None):
		current = self.player.getState()
		self.debug("update current %r", current)
		connection_manager = self.server.connection_manager_server
		av_transport = self.server.av_transport_server
		conn_id = connection_manager.lookup_avt_id(self.current_connection_id)

		if current in ( self.STATE_PLAYING, self.STATE_PAUSED ):
			self._update_transport_state(current)
		elif self.playcontainer != None and message == self.MESSAGE_EOF and \
			self.playcontainer[0] + 1 < len(self.playcontainer[2]):

			self._transition()
			next_track = ()
			item = self.playcontainer[2][self.playcontainer[0] + 1]
			infos = connection_manager.get_variable('SinkProtocolInfo')
			local_protocol_infos = infos.value.split(',')
			res = item.res.get_matching(local_protocol_infos)
			if len(res) > 0:
				res = res[0]
				infos = res.protocolInfo.split(':')
				remote_protocol, remote_network, remote_content_format, _ = infos
				didl = DIDLLite.DIDLElement()
				didl.addItem(item)
				next_track = (res.data, didl.toString(), remote_content_format)
				self.playcontainer[0] = self.playcontainer[0] + 1

			self.info("update: next=%s" %next_track)
			if len(next_track) == 3:
				av_transport.set_variable(conn_id, 'CurrentTrack',
											self.playcontainer[0] + 1)
				self.load(next_track[0], next_track[1], next_track[2])
				self.play()

			else:
				self._update_transport_state(self.STATE_IDLE)
		elif message == self.MESSAGE_EOF and \
			len(av_transport.get_variable('NextAVTransportURI').value) > 0:
			self._transition()

			CurrentURI = av_transport.get_variable('NextAVTransportURI').value
			metadata = av_transport.get_variable('NextAVTransportURIMetaData')
			CurrentURIMetaData = metadata.value
			av_transport.set_variable(conn_id, 'NextAVTransportURI', '')
			av_transport.set_variable(conn_id, 'NextAVTransportURIMetaData', '')
			r = self.upnp_SetAVTransportURI(self, InstanceID=0,
											CurrentURI=CurrentURI,
											CurrentURIMetaData=CurrentURIMetaData)
			self.info("update: r=%s" %r)
			if r == {}:
				self.play()
			else:
				self._update_transport_state(self.STATE_IDLE)
		else:
			self._update_transport_state(self.STATE_IDLE)

		self._update_transport_position()

	def _transition(self):
		self.player.transition()
		self._update_transport_state(self.STATE_TRANSITIONING)

	def _update_transport_state(self, state):
		self.info('_update_transport_state: %s' %state)
		state = {
			self.STATE_IDLE : 'STOPPED',
			self.STATE_PAUSED : 'PAUSED_PLAYBACK',
			self.STATE_PLAYING : 'PLAYING',
			self.STATE_TRANSITIONING : 'TRANSITIONING'}.get(state)

		conn_id = self.server.connection_manager_server.lookup_avt_id(self.current_connection_id)
		self.server.av_transport_server.set_variable(conn_id, 'TransportState', state)

	def _update_transport_position(self):
		connection_manager = self.server.connection_manager_server
		avt = self.server.av_transport_server
		conn_id = connection_manager.lookup_avt_id(self.current_connection_id)

		position = self._format_time(self.player.getPosition())
		duration = self._format_time(self.player.getLength())

		avt.set_variable(conn_id, 'CurrentTrackDuration', duration)
		avt.set_variable(conn_id, 'CurrentMediaDuration', duration)
		avt.set_variable(conn_id, 'RelativeTimePosition', position)
		avt.set_variable(conn_id, 'AbsoluteTimePosition', position)

		if self.metadata != None and len(self.metadata) > 0:
			if self.server != None:
				avt.set_variable(conn_id,
											'AVTransportURIMetaData',
											self.metadata)
				avt.set_variable(conn_id,
											'CurrentTrackMetaData',
											self.metadata)

	def _format_time(self, time):
		time /= 90000
		formatted = "%d:%02d:%02d" %(time/3600, time%3600/60, time%60)
		return formatted

	def load(self, uri, metadata, mimetype=None):
		self.info("loading: %r %r %r" % (uri, metadata, mimetype))

		elt = DIDLLite.DIDLElement.fromString(metadata)
		meta = None
		if(len(elt.getItems()) > 0):
			meta = Item.getItemMetadata(elt.getItems()[0])
		self.player.load(uri, meta, mimetype)
		connection_id = self.server.connection_manager_server.lookup_avt_id(self.current_connection_id)
		self.stop(silent=True) # the check whether a stop is really needed is done inside stop

		if mimetype is None:
			_, ext = os.path.splitext(uri)
			if ext == '.ogg':
				mimetype = 'application/ogg'
			else:
				mimetype = 'audio/mpeg'

		self.metadata = metadata
		self.mimetype = mimetype
		self.tags = {}

		if self.playcontainer == None:
			self.server.av_transport_server.set_variable(connection_id, 'AVTransportURI', uri)
			self.server.av_transport_server.set_variable(connection_id, 'AVTransportURIMetaData', metadata)
			self.server.av_transport_server.set_variable(connection_id, 'NumberOfTracks', 1)
			self.server.av_transport_server.set_variable(connection_id, 'CurrentTrack', 1)
		else:
			self.server.av_transport_server.set_variable(connection_id, 'AVTransportURI', self.playcontainer[1])
			self.server.av_transport_server.set_variable(connection_id, 'NumberOfTracks', len(self.playcontainer[2]))
			self.server.av_transport_server.set_variable(connection_id, 'CurrentTrack', self.playcontainer[0] + 1)

		self.server.av_transport_server.set_variable(connection_id, 'CurrentTrackURI', uri)
		self.server.av_transport_server.set_variable(connection_id, 'CurrentTrackMetaData', metadata)

		if uri.startswith('http://'):
			transport_actions = Set(['PLAY,STOP,PAUSE'])
		else:
			transport_actions = Set(['PLAY,STOP,PAUSE,SEEK'])

		if len(self.server.av_transport_server.get_variable('NextAVTransportURI').value) > 0:
			transport_actions.add('NEXT')

		if self.playcontainer != None:
			if len(self.playcontainer[2]) - (self.playcontainer[0] + 1) > 0:
				transport_actions.add('NEXT')
			if self.playcontainer[0] > 0:
				transport_actions.add('PREVIOUS')

		self.server.av_transport_server.set_variable(connection_id, 'CurrentTransportActions', transport_actions)
		self.update()

	def start(self, uri):
		self.load(uri)
		self.play()

	def stop(self, silent=False):
		self.info('Stopping: %r' % self.player.getUri())
		if not self.player.getUri() or self.player.getState() == UPnPMediaRenderer.STATE_IDLE:
			return
		self.player.stop()
		if silent is True:
			self.server.av_transport_server.set_variable(self.server.connection_manager_server.lookup_avt_id(self.current_connection_id), 'TransportState', 'STOPPED')

	def play(self):
		self.info("Playing: %r" % self.player.getUri())
		if self.player.getUri() == None:
			return
		self.player.play()
		self.server.av_transport_server.set_variable(self.server.connection_manager_server.lookup_avt_id(self.current_connection_id), 'TransportState', 'PLAYING')

	def pause(self):
		self.info('Pausing: %r' % self.player.getUri())
		self.player.pause()
		self.server.av_transport_server.set_variable(self.server.connection_manager_server.lookup_avt_id(self.current_connection_id), 'TransportState', 'PAUSED_PLAYBACK')

	def seek(self, location, old_state):
		self.player.seek(location)
		if old_state != None:
			self.server.av_transport_server.set_variable(0, 'TransportState', old_state)

	def mute(self):
		self.player.mute()
		rcs_id = self.server.connection_manager_server.lookup_rcs_id(self.current_connection_id)
		self.server.rendering_control_server.set_variable(rcs_id, 'Mute', 'True')

	def unmute(self):
		self.player.unmute()
		rcs_id = self.server.connection_manager_server.lookup_rcs_id(self.current_connection_id)
		self.server.rendering_control_server.set_variable(rcs_id, 'Mute', 'False')

	def get_mute(self):
		return self.player.getMute()

	def get_volume(self):
		return self.player.getVolume()

	def set_volume(self, volume):
		self.player.setVolume(volume)
		rcs_id = self.server.connection_manager_server.lookup_rcs_id(self.current_connection_id)
		self.server.rendering_control_server.set_variable(rcs_id, 'Volume', volume)

	def playcontainer_browse(self, uri):
		"""
		dlna-playcontainer://uuid%3Afe814e3e-5214-4c24-847b-383fb599ff01?sid=urn%3Aupnp-org%3AserviceId%3AContentDirectory&cid=1441&fid=1444&fii=0&sc=&md=0
		"""
		from urllib import unquote
		from cgi import parse_qs

		def handle_reply(r, uri, action, kw):
			try:
				next_track = ()
				elt = DIDLLite.DIDLElement.fromString(r['Result'])
				item = elt.getItems()[0]
				local_protocol_infos = self.server.connection_manager_server.get_variable('SinkProtocolInfo').value.split(',')
				res = item.res.get_matching(local_protocol_infos, protocol_type='internal')
				if len(res) == 0:
					res = item.res.get_matching(local_protocol_infos)
				if len(res) > 0:
					res = res[0]
					remote_protocol, remote_network, remote_content_format, _ = res.protocolInfo.split(':')
					didl = DIDLLite.DIDLElement()
					didl.addItem(item)
					next_track = (res.data, didl.toString(), remote_content_format)
				""" a list with these elements:

					the current track index
					- will change during playback of the container items
					the initial complete playcontainer-uri
					a list of all the items in the playcontainer
					the action methods to do the Browse call on the device
					the kwargs for the Browse call
					- kwargs['StartingIndex'] will be modified during further Browse requests
				"""
				self.playcontainer = [int(kw['StartingIndex']), uri, elt.getItems()[:], action, kw]

				def browse_more(starting_index, number_returned, total_matches):
					self.info("browse_more", starting_index, number_returned, total_matches)
					try:

						def handle_error(r):
							pass

						def handle_reply(r, starting_index):
							elt = DIDLLite.DIDLElement.fromString(r['Result'])
							self.playcontainer[2] += elt.getItems()[:]
							browse_more(starting_index, int(r['NumberReturned']), int(r['TotalMatches']))

						if((number_returned != 5 or
							number_returned < (total_matches - starting_index)) and
							(total_matches - number_returned) != starting_index):
							self.info("seems we have been returned only a part of the result")
							self.info("requested %d, starting at %d" % (5, starting_index))
							self.info("got %d out of %d" % (number_returned, total_matches))
							self.info("requesting more starting now at %d" % (starting_index + number_returned))
							self.playcontainer[4]['StartingIndex'] = str(starting_index + number_returned)
							d = self.playcontainer[3].call(**self.playcontainer[4])
							d.addCallback(handle_reply, starting_index + number_returned)
							d.addErrback(handle_error)
					except:
						import traceback
						traceback.print_exc()

				browse_more(int(kw['StartingIndex']), int(r['NumberReturned']), int(r['TotalMatches']))

				if len(next_track) == 3:
					return next_track
			except:
				import traceback
				traceback.print_exc()

			return failure.Failure(errorCode(714))

		def handle_error(r):
			return failure.Failure(errorCode(714))

		try:
			udn, args = uri[21:].split('?')
			udn = unquote(udn)
			args = parse_qs(args)

			type = args['sid'][0].split(':')[-1]

			try:
				sc = args['sc'][0]
			except:
				sc = ''

			device = self.server.coherence.get_device_with_id(udn)
			service = device.get_service_by_type(type)
			action = service.get_action('Browse')

			kw = {'ObjectID':args['cid'][0],
					'BrowseFlag':'BrowseDirectChildren',
					'StartingIndex':args['fii'][0],
					'RequestedCount':str(5),
					'Filter':'*',
					'SortCriteria':sc}

			d = action.call(**kw)
			d.addCallback(handle_reply, uri, action, kw)
			d.addErrback(handle_error)
			return d
		except:
			return failure.Failure(errorCode(714))


	def upnp_init(self):
		self.current_connection_id = None
		self.server.connection_manager_server.set_variable(0, 'SinkProtocolInfo',
							['http-get:*:application/ogg:*',
							'http-get:*:audio/mpeg:*',
							'http-get:*:audio/mp4:*',
							'http-get:*:audio/ogg:*',
							'http-get:*:audio/flac:*',
							'http-get:*:audio/x-flac:*',
							'http-get:*:audio/x-matroska:*',
							'http-get:*:audio/x-mkv:*',
							'http-get:*:audio/x-wav:*',
							'http-get:*:audio/x-wma:*',
							'http-get:*:video/mp4:*',
							'http-get:*:video/mpeg:*',
							'http-get:*:video/quicktime:*',
							'http-get:*:video/x-matroska:*',
							'http-get:*:video/x-mkv:*',
							'http-get:*:video/x-wmv:*',
							'http-get:*:video/avi:*',
							'http-get:*:video/divx:*',
							'http-get:*:image/gif:*',
							'http-get:*:image/jpeg:*',
							'http-get:*:image/png:*',
							'http-get:*:*:*'],
							default=True)
		self.server.av_transport_server.set_variable(0, 'TransportState', 'NO_MEDIA_PRESENT', default=True)
		self.server.av_transport_server.set_variable(0, 'TransportStatus', 'OK', default=True)
		self.server.av_transport_server.set_variable(0, 'CurrentPlayMode', 'NORMAL', default=True)
		self.server.av_transport_server.set_variable(0, 'CurrentTransportActions', '', default=True)
		self.server.rendering_control_server.set_variable(0, 'Volume', self.get_volume())
		self.server.rendering_control_server.set_variable(0, 'Mute', self.get_mute())

	def upnp_Play(self, *args, **kwargs):
		self.play()
		return {}

	def upnp_Pause(self, *args, **kwargs):
		self.pause()
		return {}

	def upnp_Stop(self, *args, **kwargs):
		self.stop()
		return {}

	def upnp_Seek(self, *args, **kwargs):
		InstanceID = int(kwargs['InstanceID'])
		Unit = kwargs['Unit']
		Target = kwargs['Target']
		if InstanceID != 0:
			return failure.Failure(errorCode(718))
		if Unit in ['ABS_TIME', 'REL_TIME']:
			old_state = self.server.av_transport_server.get_variable('TransportState').value
			self.server.av_transport_server.set_variable(InstanceID, 'TransportState', 'TRANSITIONING')

			sign = ''
			if Target[0] == '+':
				Target = Target[1:]
				sign = '+'
			if Target[0] == '-':
				Target = Target[1:]
				sign = '-'

			h, m, s = Target.split(':')
			seconds = int(h) * 3600 + int(m) * 60 + int(s)
			self.seek(sign + str(seconds), old_state)
		if Unit in ['TRACK_NR']:
			if self.playcontainer == None:
				NextURI = self.server.av_transport_server.get_variable('NextAVTransportURI', InstanceID).value
				if NextURI != '':
					self.server.av_transport_server.set_variable(InstanceID, 'TransportState', 'TRANSITIONING')
					NextURIMetaData = self.server.av_transport_server.get_variable('NextAVTransportURIMetaData').value
					self.server.av_transport_server.set_variable(InstanceID, 'NextAVTransportURI', '')
					self.server.av_transport_server.set_variable(InstanceID, 'NextAVTransportURIMetaData', '')
					r = self.upnp_SetAVTransportURI(self, InstanceID=InstanceID, CurrentURI=NextURI, CurrentURIMetaData=NextURIMetaData)
					return r
			else:
				Target = int(Target)
				if 0 < Target <= len(self.playcontainer[2]):
					self.server.av_transport_server.set_variable(InstanceID, 'TransportState', 'TRANSITIONING')
					next_track = ()
					item = self.playcontainer[2][Target - 1]
					local_protocol_infos = self.server.connection_manager_server.get_variable('SinkProtocolInfo').value.split(',')
					res = item.res.get_matching(local_protocol_infos)
					if len(res) > 0:
						res = res[0]
						remote_protocol, remote_network, remote_content_format, _ = res.protocolInfo.split(':')
						didl = DIDLLite.DIDLElement()
						didl.addItem(item)
						next_track = (res.data, didl.toString(), remote_content_format)
						self.playcontainer[0] = Target - 1

					if len(next_track) == 3:
						self.server.av_transport_server.set_variable(self.server.connection_manager_server.lookup_avt_id(self.current_connection_id), 'CurrentTrack', Target)
						self.load(next_track[0], next_track[1], next_track[2])
						self.play()
						return {}
			return failure.Failure(errorCode(711))

		return {}

	def upnp_Next(self, *args, **kwargs):
		InstanceID = int(kwargs['InstanceID'])
		track_nr = self.server.av_transport_server.get_variable('CurrentTrack')
		return self.upnp_Seek(self, InstanceID=InstanceID, Unit='TRACK_NR', Target=str(int(track_nr.value) + 1))

	def upnp_Previous(self, *args, **kwargs):
		InstanceID = int(kwargs['InstanceID'])
		track_nr = self.server.av_transport_server.get_variable('CurrentTrack')
		return self.upnp_Seek(self, InstanceID=InstanceID, Unit='TRACK_NR', Target=str(int(track_nr.value) - 1))

	def upnp_SetNextAVTransportURI(self, *args, **kwargs):
		NextURI = kwargs['NextURI']
		current_connection_id = self.server.connection_manager_server.lookup_avt_id(self.current_connection_id)
		NextMetaData = kwargs['NextURIMetaData']
		self.server.av_transport_server.set_variable(current_connection_id, 'NextAVTransportURI', NextURI)
		self.server.av_transport_server.set_variable(current_connection_id, 'NextAVTransportURIMetaData', NextMetaData)
		if len(NextURI) == 0  and self.playcontainer == None:
			transport_actions = self.server.av_transport_server.get_variable('CurrentTransportActions').value
			transport_actions = Set(transport_actions.split(','))
			try:
				transport_actions.remove('NEXT')
				self.server.av_transport_server.set_variable(current_connection_id, 'CurrentTransportActions', transport_actions)
			except KeyError:
				pass
			return {}
		transport_actions = self.server.av_transport_server.get_variable('CurrentTransportActions').value
		transport_actions = Set(transport_actions.split(','))
		transport_actions.add('NEXT')
		self.server.av_transport_server.set_variable(current_connection_id, 'CurrentTransportActions', transport_actions)
		return {}

	def upnp_SetAVTransportURI(self, *args, **kwargs):
		CurrentURI = kwargs['CurrentURI']
		CurrentURIMetaData = kwargs['CurrentURIMetaData']
		#print "upnp_SetAVTransportURI",InstanceID, CurrentURI, CurrentURIMetaData
		if CurrentURI.startswith('dlna-playcontainer://'):
			def handle_result(r):
				self.load(r[0], r[1], mimetype=r[2])
				return {}

			def pass_error(r):
				return r

			d = defer.maybeDeferred(self.playcontainer_browse, CurrentURI)
			d.addCallback(handle_result)
			d.addErrback(pass_error)
			return d
		elif len(CurrentURIMetaData) == 0:
			self.playcontainer = None
			self.load(CurrentURI, CurrentURIMetaData)
			return {}
		else:
			local_protocol_infos = self.server.connection_manager_server.get_variable('SinkProtocolInfo').value.split(',')
			#print local_protocol_infos
			elt = DIDLLite.DIDLElement.fromString(CurrentURIMetaData)
			if elt.numItems() == 1:
				item = elt.getItems()[0]
				res = item.res.get_matching(local_protocol_infos, protocol_type='internal')
				if len(res) == 0:
					res = item.res.get_matching(local_protocol_infos)
				if len(res) > 0:
					res = res[0]
					remote_protocol, remote_network, remote_content_format, _ = res.protocolInfo.split(':')
					self.playcontainer = None
					self.load(res.data, CurrentURIMetaData, mimetype=remote_content_format)
					return {}
		return failure.Failure(errorCode(714))

	def upnp_SetMute(self, *args, **kwargs):
		DesiredMute = kwargs['DesiredMute']
		if DesiredMute in ['TRUE', 'True', 'true', '1', 'Yes', 'yes']:
			self.mute()
		else:
			self.unmute()
		return {}

	def upnp_SetVolume(self, *args, **kwargs):
		DesiredVolume = int(kwargs['DesiredVolume'])
		self.set_volume(DesiredVolume)
		return {}

def restartMediaRenderer(session, player, name, uuid, **kwargs):
	cp = resourcemanager.getResource("UPnPControlPoint")
	if cp:
		removeUPnPDevice(uuid, cp)
		return cp.registerRenderer(UPnPMediaRenderer, session=session, player=player, name=name, uuid=uuid, **kwargs)
	return None
