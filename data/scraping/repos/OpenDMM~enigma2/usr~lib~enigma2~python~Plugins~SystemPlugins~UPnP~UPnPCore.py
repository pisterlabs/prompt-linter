# -*- coding: UTF-8 -*-
from Components.ResourceManager import resourcemanager
from Components.PluginComponent import plugins

from coherence.base import Coherence
from coherence.upnp.core import DIDLLite
from coherence.upnp.devices.control_point import ControlPoint
from coherence.upnp.devices.media_renderer import MediaRenderer
from coherence.upnp.devices.media_server import MediaServer
from coherence.upnp.devices.media_server_client import MediaServerClient
from HTMLParser import HTMLParser
from Plugins.Plugin import PluginDescriptor
from Tools.Log import Log


class Statics:
	CONTAINER_ID_ROOT = 0
	CONTAINER_ID_SERVERLIST = -1

	ITEM_TYPE_AUDIO = "audio"
	ITEM_TYPE_CONTAINER = "container"
	ITEM_TYPE_PICTURE = "picture"
	ITEM_TYPE_SERVER = "server"
	ITEM_TYPE_FILE = "file"
	ITEM_TYPE_VIDEO = "video"

	META_ALBUM = 'album'
	META_ALBUM_ART_URI = 'album_art_uri'
	META_ARTIST = 'artist'
	META_BITRATE = 'bitrate'
	META_CHILD_COUNT = 'child_count'
	META_DATE = 'date'
	META_DURATION = 'duration'
	META_GENRE = 'genre'
	META_METATYPE = 'metatype'
	META_RESOLUTION = 'resolution'
	META_SIZE = 'size'
	META_TITLE = 'title'
	META_TYPE = 'type'
	META_URI = 'uri'
	META_COVER_URI = 'cover_uri'

	SORT_TITLE_ASC = "+dc:title"
	SORT_TITLE_DSC = "+dc:title"


def hs(intval):
	"""
	Convert an int into a 2-character hex string without the 0x prepended
	e.g.: 1 -> 01
	"""
	return "%0.2X" %(intval)

class DLNA(object):
	"""
	DLNA.ORG_CI: conversion indicator parameter (boolean integer)
	0 not transcoded
	1 transcoded
	"""
	DLNA_ORG_CONVERSION_KEY = "DLNA_ORG.CI"
	DLNA_ORG_CONVERSION_NONE = 0,
	DLNA_ORG_CONVERSION_TRANSCODED = 1,

	"""
	DLNA.ORG_OP: operations parameter (string)
	"00" (or "0") neither time seek range nor range supported
	"01" range supported
	"10" time seek range supported
	"11" both time seek range and range supported
	"""
	DLNA_ORG_OPERATION_KEY = "DLNA_ORG.OP"
	DLNA_ORG_OPERATION_NONE		= 0x00,
	DLNA_ORG_OPERATION_RANGE	= 0x01,
	DLNA_ORG_OPERATION_TIMESEEK	= 0x10,

	DLNA_ORG_PROFILE_NAME_KEY = "DLNA_ORG.PN"

	LUT_MIME = {
			"mp3"  : "audio/mpeg",
			"aac"  : "audio/mp4",
			"flac" : "audio/ogg",
			"mkv"  : "video/x-matroska",
			"mp4"  : "video/mp4",
			"ts"   : "video/mpeg",
			"mpg"  : "video/mpeg",
		}

	LUT_PARAMS = {
			"video/mpeg" : ";".join(["DLNA.ORG_PN=MPEG_TS_SD_EU"] + DIDLLite.simple_dlna_tags),
		}

	@staticmethod
	def getMimeType(codec, default="audio/mpeg"):
		return DLNA.LUT_MIME.get(codec, default)

	@staticmethod
	def getParams(mimetype, default="*"):
		return DLNA.LUT_PARAMS.get(mimetype, default)

'''
This is a "managed" UPnP A/V Controlpoint which eases the use of UPnP, for Browsing media or adding a Renderer
please see the helper classes (UPnPBrowser and AbstractUPnPRenderer) for more
'''
class ManagedControlPoint(object):
	DEVICE_TYPE_SATIP_SERVER = "SatIPServer"
	DEVICE_TYPE_DREAMBOX = "Dreambox"
	URI_BASE_DREAMBOX = "urn:dreambox-de:device"

	def __init__(self):
		self.coherence = None
		self._controlPoint = None
		self.__mediaServerClients = {}
		self.__mediaRendererClients = {}
		self.__mediaDevices = {}
		self.__devices = []
		self.onMediaServerDetected = []
		self.onMediaServerRemoved  = []
		self.onMediaRendererDetected = []
		self.onMediaRendererRemoved = []
		self.onMediaDeviceDectected = []
		self.onMediaDeviceRemoved = []
		self.onSatIpServerDetected = []
		self.onSatIpServerRemoved = []
		self.onDreamboxDetected = []
		self.onDreamboxRemoved = []
		self._session = None
		self.__deferredShutDown = None
		self._startPending = False

	def _onShutdownFinished(self, *args, **kwargs):
		self.__deferredShutDown = None
		if self._startPending:
			self.start()

	def start(self):
		def doStart(*args, **kwargs):
			if self._controlPoint:
				Log.w("already running!")
				return
			Log.i("starting now!")
			self._startPending = False
			self.coherence = Coherence({
				'logging': {
					'level' : 'warning', 
					'subsystem' : [
						{'name' : 'msearch', 'level' : 'warning'},
						{'name' : 'ssdp', 'level' : 'warning'}
					]}
				})
			self._controlPoint = ControlPoint(self.coherence, auto_client=['MediaServer','MediaRenderer'])
			self.coherence.ctrl = self._controlPoint
			self.__mediaServerClients = {}
			self.__mediaRendererClients = {}
			self.__mediaDevices = {}
			self.__devices = []
			self._controlPoint.connect(self._onMediaServerDetected, 'Coherence.UPnP.ControlPoint.MediaServer.detected')
			self._controlPoint.connect(self._onMediaServerRemoved, 'Coherence.UPnP.ControlPoint.MediaServer.removed')
			self._controlPoint.connect(self._onMediaRendererDetected, 'Coherence.UPnP.ControlPoint.MediaRenderer.detected')
			self._controlPoint.connect(self._onMediaRendererRemoved, 'Coherence.UPnP.ControlPoint.MediaRenderer.removed')
			self._controlPoint.connect(self._onMediaDeviceDectected, 'Coherence.UPnP.Device.detection_completed')
			self._controlPoint.connect(self._onMediaDeviceRemoved, 'Coherence.UPnP.RootDevice.removed')
			self.__deferredShutDown = None
			if self._session:
				self._callPlugins(reason=0)
		if self.__deferredShutDown:
			Log.w("deferring start until shutdown is finished")
			if not self._startPending:
				self._startPending = True
		else:
			doStart()

	def restart(self):
		Log.i()
		if not self.__deferredShutDown:
			self.shutdown()
		self.start()

	def setSession(self, session):
		self._session = session
		if self.coherence:
			self._callPlugins(reason=0)

	def _callPlugins(self, reason=0):
		for plugin in plugins.getPlugins(PluginDescriptor.WHERE_UPNP):
			plugin(reason, session=self._session)

	def _onMediaServerDetected(self, client, udn):
		print "[DLNA] MediaServer Detected: %s (%s)" % (client.device.get_friendly_name(), client.device.get_friendly_device_type())
		self.__mediaServerClients[udn] = client
		for fnc in self.onMediaServerDetected:
			fnc(udn, client)

	def _onMediaServerRemoved(self, udn):
		if self.__mediaServerClients.get(udn, None) != None:
			del self.__mediaServerClients[udn]
			for fnc in self.onMediaServerRemoved:
				fnc(udn)

	def _onMediaRendererDetected(self, client, udn):
		print "[DLNA] MediaRenderer detected: %s (%s, %s)" % (client.device.get_friendly_name(), client.device.get_friendly_device_type(), udn)
		self.__mediaRendererClients[udn] = client
		for fnc in self.onMediaRendererDetected:
			fnc(udn, client)

	def _onMediaRendererRemoved(self, udn):
		print "[DLNA] MediaRenderer removed: %s" % (udn)
		if self.__mediaRendererClients.get(udn, None) != None:
			del self.__mediaRendererClients[udn]
			for fnc in self.onMediaRendererRemoved:
				fnc(udn)

	def _onMediaDeviceDectected(self, device):
		if device.udn in self.__mediaDevices:
			return
		self.__mediaDevices[device.udn] = device
		device_type = device.get_friendly_device_type()
		if device_type == self.DEVICE_TYPE_SATIP_SERVER:
			Log.i("New SAT>IP Server found: %s (%s - %s)" %(device.get_friendly_name(), device.get_friendly_device_type(), device.get_satipcap()))
			for fnc in self.onSatIpServerDetected:
				fnc(device)
		elif device_type == self.DEVICE_TYPE_DREAMBOX:
			Log.i("New Dreambox found: %s (%s - %s)" %(device.get_friendly_name(), device.get_friendly_device_type(), device.get_presentation_url()))
			for fnc in self.onDreamboxDetected:
				fnc(device)
		else:
			Log.i("New Device found: %s (%s)" % (device.get_friendly_name(), device.get_friendly_device_type()))

	def _onMediaDeviceRemoved(self, usn):
		if usn in self.__mediaDevices:
			print "[DLNA] Device removed: %s" % (usn)
			device = self.__mediaDevices[usn]
			device_type = device.get_friendly_device_type()
			if device_type == self.DEVICE_TYPE_SATIP_SERVER:
				for fnc in self.onSatIpServerRemoved:
					fnc(device)
			elif device_type == self.DEVICE_TYPE_DREAMBOX:
				for fnc in self.onDreamboxRemoved:
					fnc(device)
			for fnc in self.onMediaDeviceRemoved:
				fnc(device)
			del self.__mediaDevices[usn]

	def registerRenderer(self, classDef, **kwargs):
		renderer = MediaRenderer(self.coherence, classDef, no_thread_needed=True, **kwargs)
		self.__devices.append(renderer)
		return renderer

	def registerServer(self, classDef, **kwargs):
		server = MediaServer(self.coherence, classDef, no_thread_needed=True, **kwargs)
		self.__devices.append(server)
		return server

	def registerDevice(self, instance, **kwargs):
		self.__devices.append(instance)
		return instance

	def getServerList(self):
		return self.__mediaServerClients.values()

	def getRenderingControlClientList(self):
		return self.__mediaRendererClients.values()

	def getDeviceName(self, client):
		return Item.ue(client.device.get_friendly_name())

	def getSatIPDevices(self):
		devices = []
		for device in self.__mediaDevices.itervalues():
			if device.get_friendly_device_type() == self.DEVICE_TYPE_SATIP_SERVER:
				devices.append(device)
		return devices

	def getDreamboxes(self):
		devices = []
		for device in self.__mediaDevices.itervalues():
			if device.get_friendly_device_type() == self.DEVICE_TYPE_DREAMBOX:
				devices.append(device)
		return devices

	def getDevice(self, uuid):
		for device in self.__devices:
			if device.uuid == uuid:
				return device
		return None

	def removeDevice(self, uuid):
		device = self.getDevice(uuid)
		if device:
			device.unregister()
			self.__devices.remove(device)
			return True
		return False

	def shutdown(self):
		Log.i("%s" %(self.coherence,))
		if True:
			Log.w("shutdown is broken... will continue running. please restart enigma2 instead!")
			return
		if self.coherence:
			self._callPlugins(reason=1)
			self.__mediaServerClients = {}
			self.__mediaRendererClients = {}
			self.__mediaDevices = {}
			self.__devices = []
			self.__deferredShutDown = self.coherence.shutdown(force=True)
			self.__deferredShutDown.addCallback(self._onShutdownFinished)
			self._controlPoint.disconnect(self._onMediaServerDetected, 'Coherence.UPnP.ControlPoint.MediaServer.detected')
			self._controlPoint.disconnect(self._onMediaServerRemoved, 'Coherence.UPnP.ControlPoint.MediaServer.removed')
			self._controlPoint.disconnect(self._onMediaRendererDetected, 'Coherence.UPnP.ControlPoint.MediaRenderer.detected')
			self._controlPoint.disconnect(self._onMediaRendererRemoved, 'Coherence.UPnP.ControlPoint.MediaRenderer.removed')
			self._controlPoint.disconnect(self._onMediaDeviceDectected, 'Coherence.UPnP.Device.detection_completed')
			self._controlPoint.disconnect(self._onMediaDeviceRemoved, 'Coherence.UPnP.RootDevice.removed')
			self.coherence = None
			self._controlPoint = None

class Item(object):
	htmlparser = HTMLParser()

	@staticmethod
	def ue(val):
		return Item.htmlparser.unescape(val).encode("utf-8")

	@staticmethod
	def getItemType(item):
		if item != None:
			if item.__class__.__name__ == MediaServerClient.__name__:
				return Statics.ITEM_TYPE_SERVER

			itemClass = Item.ue(item.upnp_class)
			if Item.isContainer(item):
				return Statics.ITEM_TYPE_CONTAINER

			elif itemClass.startswith("object.item"):
				typeList = item.upnp_class.split('.')
				if "videoItem" in typeList or "movie" in typeList:
					return Statics.ITEM_TYPE_VIDEO
				elif "musicTrack" in typeList or "audioItem" in typeList:
					return Statics.ITEM_TYPE_AUDIO
				elif "photo" in typeList:
					return Statics.ITEM_TYPE_PICTURE
				else:
					return Statics.ITEM_TYPE_FILE
		return None

	@staticmethod
	def isServer(item):
		return Item.getItemType(item) == Statics.ITEM_TYPE_SERVER

	@staticmethod
	def getServerName(client):
		return Item.ue(client.device.get_friendly_name())

	'''
	Returns the title of the current item
	'''
	@staticmethod
	def getItemTitle(item):
		if Item.isServer(item):
			return Item.getServerName(item)

		if item.title != None:
			return Item.ue(item.title)
		else:
			return "<missing title>"

	'''
	returns the number of children for container items
	returns -1 for non-container items
	'''
	@staticmethod
	def getItemChildCount(item):
		if Item.getItemType(item) != Statics.ITEM_TYPE_SERVER and Item.isContainer(item):
			return item.childCount

		return -1

	'''
	Currently always returns a dict with the first url and meta-type, which is usually the original/non-transcoded source
	Raises an IllegalInstanceException if you pass in a container-item
	'''
	@staticmethod
	def getItemUriMeta(item):
		assert not Item.isContainer(item)
		for res in item.res:
			uri = Item.ue(res.data)
			meta = Item.ue(res.protocolInfo.split(":")[2])
			print "URL: %s\nMeta:%s" %(uri, meta)
			if uri:
				return {Statics.META_URI : uri, Statics.META_METATYPE : meta}

	@staticmethod
	def getItemId(item):
		if Item.isServer(item):
			return item.device.get_id()
		else:
			return item.id

	@staticmethod
	def getAttrOrDefault(instance, attr, default=None):
		val = getattr(instance, attr, default) or default
		try:
			return Item.ue(val)
		except:
			return val

	@staticmethod
	def getItemMetadata(item):
		type = Item.getItemType(item)
		meta = {}
		metaNA = _('n/a')
		cover_uri = None

		if type == Statics.ITEM_TYPE_SERVER:
			meta = {
					Statics.META_TYPE : type,
					Statics.META_TITLE : Item.getServerName(item),
				}

		elif type == Statics.ITEM_TYPE_CONTAINER:
			meta = {
					Statics.META_TYPE : type,
					Statics.META_TITLE : Item.getAttrOrDefault(item, 'title', metaNA),
					Statics.META_CHILD_COUNT : Item.getItemChildCount(item),
				}
		elif type == Statics.ITEM_TYPE_PICTURE or type == Statics.ITEM_TYPE_VIDEO:
			for res in item.res:
				content_format = Item.ue( res.protocolInfo.split(':')[2] )
				if ( type == Statics.ITEM_TYPE_VIDEO and content_format.startswith("video") ) or type == Statics.ITEM_TYPE_PICTURE:
					meta = {
							Statics.META_TYPE : type,
							Statics.META_METATYPE : content_format,
							Statics.META_TITLE : Item.getAttrOrDefault(item, 'title', metaNA),
							Statics.META_DATE : Item.getAttrOrDefault(item, 'date', metaNA),
							Statics.META_RESOLUTION : Item.getAttrOrDefault(item, 'resolution', metaNA),
							Statics.META_SIZE : Item.getAttrOrDefault(item, 'size', -1),
							Statics.META_URI : Item.getAttrOrDefault(res, 'data'),
						}
				elif type == Statics.ITEM_TYPE_VIDEO and content_format.startswith("image"):
					cover_uri = Item.getAttrOrDefault(res, 'data')

				if type == Statics.ITEM_TYPE_PICTURE:
					meta[Statics.META_ALBUM] = Item.getAttrOrDefault(item, 'album', metaNA)
				elif type == Statics.ITEM_TYPE_VIDEO:
					meta[Statics.META_ALBUM_ART_URI] = Item.getAttrOrDefault(item, 'albumArtURI')

		elif type == Statics.ITEM_TYPE_AUDIO:
			for res in item.res:
				content_format = Item.ue( res.protocolInfo.split(':')[2] )
				if content_format.startswith("audio"):
					meta = {
							Statics.META_TYPE : type,
							Statics.META_METATYPE : content_format,
							Statics.META_TITLE : Item.getAttrOrDefault(item, 'title', metaNA),
							Statics.META_ALBUM : Item.getAttrOrDefault(item, 'album', metaNA),
							Statics.META_ARTIST : Item.getAttrOrDefault(item, 'artist', metaNA),
							Statics.META_GENRE : Item.getAttrOrDefault(item, 'genre', metaNA),
							Statics.META_DURATION : Item.getAttrOrDefault(item, 'duration', "0"),
							Statics.META_BITRATE : Item.getAttrOrDefault(item, 'bitrate', "0"),
							Statics.META_SIZE : Item.getAttrOrDefault(item, 'size', -1),
							Statics.META_ALBUM_ART_URI : Item.getAttrOrDefault(item, 'albumArtURI'),
							Statics.META_URI : Item.getAttrOrDefault(res, 'data'),
						}
				elif content_format.startswith("image"):
					cover_uri = Item.getAttrOrDefault(res, 'data')
		if cover_uri != None:
			meta[Statics.META_COVER_URI] = cover_uri

		return meta

	@staticmethod
	def isContainer(item):
		if item.__class__.__name__ == MediaServerClient.__name__:
			return True
		return item.upnp_class.startswith("object.container")

def removeUPnPDevice(uuid, cp=None):
	if not cp:
		cp = resourcemanager.getResource("UPnPControlPoint")
	if cp:
		return cp.removeDevice(uuid)
	return False
