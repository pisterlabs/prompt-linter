from __future__ import print_function
from coherence.upnp.core import DIDLLite

class UPnPMediaRenderingControlClient(object):
	STATE_STOPPED = "STOPPED"
	STATE_PAUSED_PLAYBACK = "PAUSED_PLAYBACK"
	STATE_PLAYING = "PLAYING"
	STATE_TRANSITIONING = "TRANSITIONING"
	STATE_NO_MEDIA_PRESENT = "NO_MEDIA_PRESENT"
	CONNECTION_STATE_ERROR = "ERROR"
	TRANSPORT_STATE_OK = "OK"

	UPNP_NOT_IMPLEMENTED = "NOT_IMPLEMENTED"

	def __init__(self, client):
		self.__client = client #MediaRendererClient
		self.__transport = self.__client.av_transport #AVTransportClient
		self.__renderclient = self.__client.rendering_control #RenderingControlClient
		#useless without avtransport- and/or renderering client
		assert self.__transport is not None
		assert self.__renderclient is not None

		self.onTransportStatusChanged = []
		self.onPlaybackStatusChanged = []
		self.onDurationChanged = []
		self.onPositionChanged = []

		self.__subscribe()

	def __subscribe(self):
		self.__transport.subscribe_for_variable("LastChange", self.__onStateChanged, signal=True)

	def __onStateChanged(self, variable):
		print("__onStateChanged: %s" %(variable.name))
		self.__onTransportStatusChanged(self.__transport.service.get_state_variable("TransportStatus"))
		self.__onTransportStateChanged(self.__transport.service.get_state_variable("TransportState"))
		self.__onDurationChanged(self.__transport.service.get_state_variable("CurrentTrackDuration"))
		self.__onPositionChanged(self.__transport.service.get_state_variable("AbsoluteTimePosition"))

	def __onTransportStatusChanged(self, status):
		print("__onTransportStatusChanged status=%s" %(status.value))
		for fnc in self.onTransportStatusChanged:
			fnc(status.value)

	def __onTransportStateChanged(self, state):
		print("__onTransportStateChanged state=%s" %(state.value))
		for fnc in self.onPlaybackStatusChanged:
			fnc(state.value)

	'''
	converts a upnp timestamp in the format HH:mm:ss to seconds
	e.g.: 00:01:30 -> 90
	'''
	def __tsToSecs(self, timestamp):
		val = timestamp.split(":")
		secs = (float(val[0]) * 3600) + (float(val[1]) * 60) + float(val[2])
		return secs

	def __onDurationChanged(self, duration):
		print("[UPnPMediaRenderingControlClient].__onDurationChanged, duration=%s" %duration.value)
		for fnc in self.onDurationChanged:
			fnc( self.__tsToSecs(duration.value) )

	def __onPositionChanged(self, pos):
		print("[UPnPMediaRenderingControlClient].__onPositionChanged, pos=%s" %pos.value)
		for fnc in self.onPositionChanged:
			fnc( self.__tsToSecs(pos.value) )

	def getDeviceName(self):
		return self.__client.device.get_friendly_name()

	def setMediaUri(self, uri, item):
		elt = DIDLLite.DIDLElement()
		elt.addItem(item)
		metadata = elt.toString()
		self.__transport.set_av_transport_uri(current_uri=uri, current_uri_metadata=metadata)

	def setNextMediaUri(self, uri = '', metadata = ''):
		self.__transport.set_next_av_transport_uri(next_uri=uri, next_uri_metadata=metadata)

	def playUri(self, uri = '', metadata = ''):
		self.setMediaUri(uri, metadata)
		self.play()

	def getPosition(self):
		return self.__transport.get_position_info()

	def play(self):
		self.__transport.play()

	def pause(self):
		self.__transport.pause()

	def seek(self, target):
		self.__transport.seek(target=target)

	def next(self):
		self.__transport.next()

	def prev(self):
		self.__transport.previous()

	def stop(self):
		self.__transport.stop()

	def getMute(self):
		return self.__renderclient.get_mute() == 1

	def setMute(self, mute):
		val = int(mute)
		self.__renderclient.set_mute(desired_mute=val)

	def getVolume(self):
		return self.__renderclient.get_volume()

	def setVolume(self, target):
		self.__renderclient.set_volume(desired_volume=target)

