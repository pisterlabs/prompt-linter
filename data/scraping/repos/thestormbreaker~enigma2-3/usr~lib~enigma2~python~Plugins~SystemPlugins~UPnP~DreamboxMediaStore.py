# -*- coding: utf-8 -*-

"""
This is a Media Backend that allows you to access audio, videos and live-streams on
a Dreambox.
"""

from __future__ import absolute_import
from enigma import eMediaDatabase, eServiceReference, StringList
from Components.config import config
from Components.ResourceManager import resourcemanager
from Components.Sources.ServiceList import ServiceList
from Tools.Log import Log

from coherence.backend import ROOT_CONTAINER_ID, AbstractBackendStore, Container
from coherence.upnp.core import DIDLLite
from coherence.upnp.core.soap_service import errorCode

from twisted.internet import defer

from .UPnPCore import DLNA, removeUPnPDevice

from coherence.extern.et import ET
from twisted.python import failure

import six

def convertString(string):
	return str(string).encode('utf-8', 'replace')

class RootContainer(Container):
	def get_item(self):
		if self.item is None:
			self.item = DIDLLite.Container(self.storage_id, self.parent_id, self.name)
		self.item.childCount = len(self.children)
		return self.item

class BaseContainer(RootContainer):
	def __init__(self, parent, name):
		RootContainer.__init__(self, parent, convertString(name))

	def _set_item_defaults(self, searchClass=[]):
		if self.children is None:
			self.get_children()
		if self.children:
			self.item.childCount = self.get_child_count()
		self.item.date = None
		self.item.genre = None
		self.item.searchable = True
		self.item.searchClass = searchClass

class DBContainer(BaseContainer):
	ITEM_KEY_TITLE = "title"
	ITEM_KEY_ID = "id"
	ITEM_KEY_CATEGORY = "category"

	def __init__(self, db, name, parent):
		BaseContainer.__init__(self, parent, name)
		self._db = db
		self.location = None
		self.item = None
		self.children = None
		self._default_mime = "audio/mpeg"
		self._default_additionalInfo = "*"

	def get_path(self):
		return self.location

	def get_item(self):
		if self.item is None:
			self.item = DIDLLite.Item(self.get_id(), self.parent.get_id(), self.name, restricted=True)
			self._set_item_defaults()
			self.update_id += 1
		return self.item

	def _get_data(self, res):
		if res and not res.error():
			return res.data()
		elif res:
			self.warning("%s\n%s" % (res.errorDriverText(), res.errorDatabaseText()))
		else:
			self.warning("res is %s", res)
		return []

	def _set_item_resources(self, codec, size=0, duration=0, resolution=None):
			if size < 0:
				size = 0
			mimetype = DLNA.getMimeType(codec, self._default_mime)
			dlna_params = DLNA.getParams(mimetype)

			ext = self.location.split('.')[-1]
			url = "%s%s.%s" % (self.store.urlbase, self.get_id(), ext)

			res = DIDLLite.Resource(data=url, protocolInfo='http-get:*:%s:%s' % (mimetype, dlna_params))
			res.size = size

			m, s = divmod(duration, 60)
			h, m = divmod(m, 60)
			res.duration = "%d:%02d:%02d.000" % (h, m, s)
			res.resolution = resolution
			self.item.res.append(res)

	def _set_item_defaults(self, searchClass=[]):
		BaseContainer._set_item_defaults(self, searchClass)
		self.item.genre = _("Unknown")

class Artist(DBContainer):
	schemaVersion = 1
	mimetype = 'directory'

	def __init__(self, db, name, parent, isAlbumArtist=False):
		DBContainer.__init__(self, db, name, parent)
		self._isAlbumArtist = isAlbumArtist

	def _do_get_children(self, start, end):
		return self._db.getAlbumsByArtist(self.name)

	def get_children(self, start=0, end=0):
		if self.children is None:
			res = self._do_get_children(start, end)
			items = self._get_data(res)
			self.children = []
			if len(items) > 1:
				self.add_child(ArtistAll(self._db, self.name, _("-- All --"), self, isAlbumArtist=self._isAlbumArtist))
			for item in items:
				self.add_child(Album(self._db, item, self, isAlbumArtist=self._isAlbumArtist))
		return DBContainer.get_children(self, start, end)

	def get_item(self):
#		self.warning(self.name)
		if self.item is None:
			self.item = DIDLLite.MusicArtist(self.get_id(), self.parent.get_id(), self.name, restricted=True)
			self._set_item_defaults(searchClass="object.item.audioItem")
		return self.item

class AlbumArtist(Artist):
	schemaVersion = 1
	mimetype = 'directory'

	def __init__(self, db, name, parent):
		Artist.__init__(self, db, name, parent, isAlbumArtist=True)

	def _do_get_children(self, start, end):
		return self._db.getAlbumsByAlbumArtist(self.name)

class Artists(DBContainer):
	schemaVersion = 1
	mimetype = 'directory'

	def get_children(self, start=0, end=0):
		if self.children is None:
			res = self._db.getAllArtists()
			Log.i(self.name)
			items = self._get_data(res)
			self.children = []
			for item in items:
				self.add_child(Artist(self._db, item[eMediaDatabase.FIELD_ARTIST], self))
		return DBContainer.get_children(self, start, end)

	def get_item(self):
		if self.item is None:
			self.item = DIDLLite.Music(self.get_id(), self.parent.get_id(), self.name, restricted=True)
			self._set_item_defaults(searchClass="object.item.audioItem")
		return self.item

class AlbumArtists(Artists):
	schemaVersion = 1
	mimetype = 'directory'

	def get_children(self, start=0, end=0):
		if self.children is None:
			res = self._db.getAllAlbumArtists()
			Log.i(self.name)
			items = self._get_data(res)
			self.children = []
			for item in items:
				self.add_child(AlbumArtist(self._db, item[eMediaDatabase.FIELD_ARTIST], self))
		return DBContainer.get_children(self, start, end)

class AudioAll(DBContainer):
	schemaVersion = 1
	mimetype = 'directory'

	def get_children(self, start=0, end=0):
		if self.children is None:
			Log.i(self.name)
			self.children = []
			res = self._db.getAllAudio()
			items = self._get_data(res)
			for item in items:
				self.add_child(Track(self._db, item, self, combinedTitle=True))
		return DBContainer.get_children(self, start, end)

	def get_item(self):
		Log.i(self.item)
		if self.item is None:
			self.item = DIDLLite.Music(self.get_id(), self.parent.get_id(), self.name, restricted=True)
			self._set_item_defaults(searchClass="object.item.audioItem")
		return self.item

class Album(DBContainer):
	schemaVersion = 1
	mimetype = 'directory'

	def __init__(self, db, item, parent, isAlbumArtist=False):
		artist, album, album_id = item[eMediaDatabase.FIELD_ARTIST], item[eMediaDatabase.FIELD_ALBUM], item[eMediaDatabase.FIELD_ID]
		self._db_item = item
		self.artist = artist
		self.id = int(album_id)
		self._isAlbumArtist = isAlbumArtist
		DBContainer.__init__(self, db, album, parent)

	def get_children(self, start=0, end=0):
		if self.children is None:
			self.children = []
			if self.id:
				res = self._db.getTracksByAlbumId(self.id)
			else:
				if self._isAlbumArtist:
					res = self._db.filterByAlbumArtistAlbum(self.artist, self.name)
				else:
					res = self._db.filterByArtistAlbum(self.artist, self.name)
			items = self._get_data(res)
			for item in items:
				self.add_child(Track(self._db, item, self))
		return DBContainer.get_children(self, start, end)

	def get_item(self):
		if self.item is None:
			self.item = DIDLLite.MusicAlbum(id=self.get_id(), parentID=self.parent.get_id(), title=self.name, restricted=True)
			self._set_item_defaults(searchClass="object.item.audioItem")
			self.item.artist = self.artist
			self.item.creator = self.artist
			if self.id:
				res = self._db.getAlbumCoverArtId(self.id)
				if res and not res.error():
					data = res.data()
					for d in data:
						for coverid in six.itervalues(d):
							if int(coverid) != 0:
								self.item.albumArtURI = self.store.getCoverArtUri(coverid)
#								Log.d("%s - %s: %s" %(self.artist, self.name, self.item.albumArtURI))
								break
		return self.item

class ArtistAll(Album):
	schemaVersion = 1
	mimetype = 'directory'
	def __init__(self, db, artist, name, parent, isAlbumArtist=False):
		item = {
				eMediaDatabase.FIELD_ARTIST : artist,
				eMediaDatabase.FIELD_ALBUM : name,
				eMediaDatabase.FIELD_ID : 0
			}
		Album.__init__(self, db, item, parent, isAlbumArtist=isAlbumArtist)

	def get_children(self, start=0, end=0):
		if self.children is None:
			self.children = []
			if self._isAlbumArtist:
				res = self._db.filterByAlbumArtist(self.parent.name)
			else:
				res = self._db.filterByArtist(self.parent.name)
			items = self._get_data(res)
			for item in items:
				self.add_child(Track(self._db, item, self))
		return DBContainer.get_children(self, start, end)

class Albums(DBContainer):
	schemaVersion = 1
	mimetype = 'directory'

	def get_children(self, start=0, end=0):
		if self.children is None:
			self.children = []
			res = self._db.getAllAlbums()
			items = self._get_data(res)
			for item in items:
				self.add_child(Album(self._db, item, self))
		return DBContainer.get_children(self, start, end)

	def get_item(self):
		if self.item is None:
			self.item = DIDLLite.Music(id=self.get_id(), parentID=self.parent.get_id(), title=self.name, restricted=True)
			self._set_item_defaults(searchClass="object.item.audioItem")
		return self.item

class Track(DBContainer):
	schemaVersion = 1

	def __init__(self, db, item, parent, combinedTitle=False):
		self._db_item = item
		artist, album, title, size, date, duration, genre, coverid = item[eMediaDatabase.FIELD_ARTIST], item[eMediaDatabase.FIELD_ALBUM], item[eMediaDatabase.FIELD_TITLE], item[eMediaDatabase.FIELD_SIZE], item[eMediaDatabase.FIELD_DATE], item[eMediaDatabase.FIELD_DURATION], item[eMediaDatabase.FIELD_GENRE], int(item[eMediaDatabase.FIELD_COVER_ART_ID])
		self.artist = convertString(artist)
		self.title = convertString(title)
		if combinedTitle:
			self.title = "%s - %s" %(self.artist, self.title)
		self.album = convertString(album)
		self.coverid = coverid
		self.size = convertString(size)
		self.date = convertString(date)
		self.duration = int(duration)
		self.genre = genre

		DBContainer.__init__(self, db, self.title, parent)
		self.location = "%s/%s" % (item[eMediaDatabase.FIELD_PATH], item[eMediaDatabase.FIELD_FILENAME])

	def get_children(self, start=0, request_count=0):
		return []

	def get_item(self):
		if self.item is None:
			self.item = DIDLLite.MusicTrack(self.get_id(), self.parent.get_id(), self.name, restricted=True)
			self._set_item_defaults()
			self.item.artist = self.artist
			self.item.creator = self.artist
			self.item.title = self.title
			self.item.album = self.album
			self.item.description = self.album
			self.item.date = self.date
			self.item.genre = self.genre
			self.item.restricted = True
			if self.coverid:
				self.item.albumArtURI = self.store.getCoverArtUri(self.coverid)
			codec = self._db_item['codec']
			self._set_item_resources(codec, size=self.size, duration=self.duration)
		return self.item

class VideoContainer(DBContainer):
	schemaVersion = 1
	mimetype = 'directory'

	def get_children(self, start=0, end=0):
		if self.children is None:
			self.children = []
			res = self._get_res()
			items = self._get_data(res)
			for item in items:
				self.add_child(Video(self._db, item, self))
		return DBContainer.get_children(self, start, end)

	def _get_res(self):
		return []

	def get_item(self):
		if self.item is None:
			self.item = DIDLLite.Container(id=self.get_id(), parentID=self.parent.get_id(), title=self.name, restricted=True)
			self._set_item_defaults(searchClass="object.item.videoItem")
		return self.item

class VideoRecordings(VideoContainer):
	def _get_res(self):
		return self._db.getAllRecordings()

class VideoUnseen(VideoContainer):
	def _get_res(self):
		return self._db.query("SELECT * from video WHERE lastplaypos=?;", StringList(["0"]))

class VideoHD(VideoContainer):
	def _get_res(self):
		return self._db.query("SELECT * from video WHERE hd=?;", StringList(["1"]))

class VideoSD(VideoContainer):
	def _get_res(self):
		return self._db.query("SELECT * from video WHERE hd=?;", StringList(["0"]))

class VideoAll(VideoContainer):
	def _get_res(self):
		return self._db.getAllVideos()

class Video(DBContainer):
	schemaVersion = 1

	def __init__(self, db, item, parent):
		self._db_item = item
		name, size, duration = item[eMediaDatabase.FIELD_TITLE], item[eMediaDatabase.FIELD_SIZE], item[eMediaDatabase.FIELD_DURATION]
		DBContainer.__init__(self, db, name, parent)
		self.location = "%s/%s" % (item[eMediaDatabase.FIELD_PATH], item[eMediaDatabase.FIELD_FILENAME])
		self.size = size
		self.duration = int(duration)

	def get_children(self, start=0, request_count=0):
		return []

	def get_item(self):
		if self.item is None:
			self.item = DIDLLite.Movie(self.get_id(), self.parent.get_id(), self.name, restricted=True)
			self._set_item_defaults(searchClass="object.item.videoItem")
			self.item.title = self.name
			self._default_mime = "video/mpeg"
			codec = self._db_item['codec']
			self._set_item_resources(codec, self.size, self.duration, resolution="720x576") #FIXME resolution is hardcoded
		return self.item

class DVBServiceList(BaseContainer):
	def __init__(self, parent, title, ref):
		BaseContainer.__init__(self, parent, title)
		self.service_number = 0
		self.item = None
		self.location = ""
		self.children = None
		self.ref = ref
		self.sorted = True
		def childs_sort(x,y):
			return 1
		self.sorting_method = childs_sort

	def get_service_number(self):
		return self.service_number

	def get_item(self):
		if self.item == None:
			self.item = DIDLLite.Container(self.get_id(), self.parent.get_id(), self.name, restricted=True)
			self._set_item_defaults(searchClass="object.item.videoItem")
		return self.item

	def get_children(self, start=0, end=0):
		if self.children is None:
			self.children = []
			self._init_services(self.ref)
		return BaseContainer.get_children(self, start, end)

	def _get_next_service_nr(self):
		self.service_number += 1
		return self.service_number

	def _gen_child(self, ref, name):
		return DVBServiceList(self, name, ref)

	def _init_services(self, ref):
		self.info(ref)
		servicelist = None
		def get_servicelist(ref):
			servicelist.root = ref
		if ref:
			ref = eServiceReference(ref)
			if not ref.valid():
				self.warning("Invalid ref %s" % ref)
				return []
		else:
			self.warning("Missing ref!")

		servicelist = ServiceList(ref, command_func=get_servicelist, validate_commands=False)
		services = servicelist.getServicesAsList()
		for ref, name in services:
			if ref.startswith("1:64"): #skip markers
				continue
			child = self._gen_child(ref, name)
			self.add_child(child)

class DVBService(DVBServiceList):
	def __init__(self, parent, title, ref, is_radio=False):
		DVBServiceList.__init__(self, parent, title, ref)
		self.location = None
		self.streaminghost = None

	def get_service_number(self):
		return self.service_number

	def get_path(self):
		if self.streaminghost is None:
			self.streaminghost = self.store.server.coherence.hostname
		if self.location is None:
			self.location = 'http://' + self.streaminghost + ':8001/' + self.ref
		return self.location

	def get_item(self):
		if self.item == None:
			self.item = DIDLLite.VideoBroadcast(self.get_id(), self.parent.get_id(), self.name, restricted=True)
			self._set_item_defaults(searchClass="object.item.videoItem")
			mimetype = "video/mpeg"
			additional_info = DLNA.getParams(mimetype)
			res = DIDLLite.Resource(self.get_path(), 'http-get:*:%s:%s' %(mimetype, additional_info))
			res.size = None
			self.item.res.append(res)
		return self.item

	def get_children(self, start=0, end=0):
		return []

class Favorite(DVBServiceList):
	def _gen_child(self, ref, name):
		return DVBService(self, name, ref)

class Favorites(DVBServiceList):
	def _gen_child(self, ref, name):
		return Favorite(self, name, ref)

class Provider(DVBServiceList):
	def _gen_child(self, ref, name):
		return DVBService(self, name, ref)

class ProviderList(DVBServiceList):
	def _gen_child(self, ref, name):
		return Provider(self, name, ref)

from .WebCoverResource import CoverRoot

class DreamboxMediaStore(AbstractBackendStore):
	implements = ['MediaServer']
	logCategory = 'dreambox_media_store'

	def __init__(self, server, *args, **kwargs):
		AbstractBackendStore.__init__(self, server, **kwargs)
		self._db = eMediaDatabase.getInstance()

		self.next_id = 1000
		self.name = kwargs.get('name', 'Dreambox Mediaserver')
		# streaminghost is the ip address of the dreambox machine, defaults to localhost
		self.streaminghost = kwargs.get('streaminghost', self.server.coherence.hostname)

		self.refresh = float(kwargs.get('refresh', 1)) * 60
		self.init_root()
		self.init_completed()

		self._cover_root = None

		#Samsung TVs are special...
#		self._X_FeatureList = """&lt;Features xmlns=\"urn:schemas-upnp-org:av:avs\""
#		" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\""
#		" xsi:schemaLocation=\"urn:schemas-upnp-org:av:avs http://www.upnp.org/schemas/av/avs.xsd\"&gt;"
#		" &lt;Feature name=\"samsung.com_BASICVIEW\" version=\"1\"&gt;"
#		 "&lt;container id=\"1\" type=\"object.item.audioItem\"/&gt;"
#		 "&lt;container id=\"2\" type=\"object.item.videoItem\"/&gt;"
#		 "&lt;container id=\"3\" type=\"object.item.imageItem\"/&gt;&lt;/Features&gt;"""

	def getCoverArtUri(self, cover_id):
		return "%s/cover/%s" %(self.server.urlbase, cover_id)

	def init_cover_root(self):
		if not self._cover_root:
			self._cover_root = self.server.web_resource.putChild("cover", CoverRoot())

	def init_root(self):
		root = RootContainer(None, -1)
		self.set_root_item(root)

		#AUDIO
		if config.plugins.mediaserver.share_audio.value:
			self._audio = RootContainer(root, "Audio")
			root.add_child(self._audio, ROOT_CONTAINER_ID)

			self._artists_id = self._audio.add_child(
					Artists(self._db, _("Artists"), self._audio)
				)
			self._albums_id = self._audio.add_child(Albums(
					self._db, _("Albums"), self._audio)
				)
			self._album_artists_id = self._audio.add_child(
					AlbumArtists(self._db, _("Album Artists"), self._audio)
				)
			self._audio_all_id = self._audio.add_child(
					AudioAll(self._db, _("All"), self._audio)
				)

		#VIDEO
		if config.plugins.mediaserver.share_video.value:
			self._video = RootContainer(root, "Video")
			self._video_root_id = root.add_child(self._video)

			self._video_recordings_id = self._video.add_child(
					VideoRecordings(self._db, _("Recordings"), self._video)
				)
			self._video_unseen_id = self._video.add_child(
					VideoUnseen(self._db, _("Unseen"), self._video)
				)
			self._video_hd_id = self._video.add_child(
					VideoHD(self._db, _("HD"), self._video)
				)
			self._video_sd_id = self._video.add_child(
					VideoSD(self._db, _("SD"), self._video)
				)
			self._video_all_id = self._video.add_child(
					VideoAll(self._db, _("All"), self._video)
				)
		#DVB LIVE
		if config.plugins.mediaserver.share_live.value:
			self._live = RootContainer(root, "Livestreams (DVB)")
			self._live_root_id = root.add_child(self._live)
			#TV
			self._live_favorites_tv_id = self._live.add_child(
					Favorites(self._live, _("Favorites (TV)"), "1:7:1:0:0:0:0:0:0:0:(type == 1) || (type == 17) || (type == 195) || (type == 25) FROM BOUQUET \"bouquets.tv\" ORDER BY bouquet")
				)
			self._live_provider_tv_id = self._live.add_child(
					ProviderList(self._live, _("Provider (TV)"), "1:7:1:0:0:0:0:0:0:0:(type == 1) || (type == 17) || (type == 195) || (type == 25) FROM PROVIDERS ORDER BY name")
				)
			#RADIO
			self._live_favorites_radio_id = self._live.add_child(
					Favorites(self._live, _("Favorites (Radio)"), "1:7:2:0:0:0:0:0:0:0:(type == 2)FROM BOUQUET \"bouquets.radio\" ORDER BY bouquet")
				)
			self._live_provider_radio_id = self._live.add_child(
					ProviderList(self._live, _("Provider (Radio)"), "1:7:2:0:0:0:0:0:0:0:(type == 2) FROM PROVIDERS ORDER BY name")
				)

		root.sorted = True
		def childs_sort(x, y):
			return cmp(x.name, y.name)
		root.sorting_method = childs_sort

	def upnp_GetSearchCapabilities(self, *args, **kwargs):
		self.init_cover_root()
		return {"SearchCaps" : "upnp:class,dc:title"}

	def upnp_Browse(self, *args, **kwargs):
		self.init_cover_root()
		return self.server.content_directory_server.upnp_Browse(self, *args, **kwargs)

	def upnp_Search(self, *args, **kwargs):
		self.init_cover_root()
		Log.d("%s" % (kwargs,))
		ContainerID = kwargs['ContainerID']
		StartingIndex = int(kwargs['StartingIndex'])
		RequestedCount = int(kwargs['RequestedCount'])
		SearchCriteria = kwargs['SearchCriteria']

		total = 0
		root_id = 0
		item = None
		items = []
		parent_container = None
		SearchCriteria = SearchCriteria.split(" ")
		if "derivedfrom" in SearchCriteria:
			if SearchCriteria[0] == "upnp:class":
				Log.d("Searching by class! %s" % (SearchCriteria,))
				if SearchCriteria[2].find(DIDLLite.AudioItem.upnp_class) >= 0:
					parent_container = str(self._audio_all_id)
				elif SearchCriteria[2].find(DIDLLite.VideoItem.upnp_class) >= 0:
					parent_container = str(self._video_all_id)
				Log.d(parent_container)
		if not parent_container:
			parent_container = str(ContainerID)
		didl = DIDLLite.DIDLElement(upnp_client=kwargs.get('X_UPnPClient', ''),
			parent_container=parent_container,
			transcoding=self.server.content_directory_server.transcoding)

		def build_response(tm):
			r = {'Result': didl.toString(), 'TotalMatches': tm,
				 'NumberReturned': didl.numItems()}

			if hasattr(item, 'update_id'):
				r['UpdateID'] = item.update_id
			elif hasattr(self, 'update_id'):
				r['UpdateID'] = self.update_id # FIXME
			else:
				r['UpdateID'] = 0

			return r

		def got_error(r):
			return r

		def process_result(result, total=None, found_item=None):
			if result == None:
				result = []

			l = []

			def process_items(result, tm):
				if result == None:
					result = []
				for i in result:
					if i[0] == True:
						didl.addItem(i[1])

				return build_response(tm)

			for i in result:
				d = defer.maybeDeferred(i.get_item)
				l.append(d)

			if found_item != None:
				def got_child_count(count):
					dl = defer.DeferredList(l)
					dl.addCallback(process_items, count)
					return dl

				d = defer.maybeDeferred(found_item.get_child_count)
				d.addCallback(got_child_count)

				return d
			elif total == None:
				total = item.get_child_count()

			dl = defer.DeferredList(l)
			dl.addCallback(process_items, total)
			return dl

		def proceed(result):
			if kwargs.get('X_UPnPClient', '') == 'XBox' and hasattr(result, 'get_artist_all_tracks'):
				d = defer.maybeDeferred(result.get_artist_all_tracks, StartingIndex, StartingIndex + RequestedCount)
			else:
				d = defer.maybeDeferred(result.get_children, StartingIndex, StartingIndex + RequestedCount)
			d.addCallback(process_result, found_item=result)
			d.addErrback(got_error)
			return d

		try:
			root_id = parent_container
		except:
			pass

		wmc_mapping = getattr(self, "wmc_mapping", None)
		if kwargs.get('X_UPnPClient', '') == 'XBox':
			if wmc_mapping and parent_container in wmc_mapping:
				""" fake a Windows Media Connect Server
				"""
				root_id = wmc_mapping[parent_container]
				if callable(root_id):
					item = root_id()
					if item is not None:
						if isinstance(item, list):
							total = len(item)
							if int(RequestedCount) == 0:
								items = item[StartingIndex:]
							else:
								items = item[StartingIndex:StartingIndex + RequestedCount]
							return process_result(items, total=total)
						else:
							if isinstance(item, defer.Deferred):
								item.addCallback(proceed)
								return item
							else:
								return proceed(item)

				item = self.get_by_id(root_id)
				if item == None:
					return process_result([], total=0)

				if isinstance(item, defer.Deferred):
					item.addCallback(proceed)
					return item
				else:
					return proceed(item)


		item = self.get_by_id(root_id)
		Log.w(item)
		if item == None:
			Log.w(701)
			return failure.Failure(errorCode(701))

		if isinstance(item, defer.Deferred):
			item.addCallback(proceed)
			return item
		else:
			return proceed(item)

	def upnp_init(self):
		if self.server:
			self.server.connection_manager_server.set_variable(0, 'SourceProtocolInfo', [
					'http-get:*:image/jpeg:DLNA.ORG_PN=JPEG_TN',
					'http-get:*:image/jpeg:DLNA.ORG_PN=JPEG_SM',
					'http-get:*:image/jpeg:DLNA.ORG_PN=JPEG_MED',
					'http-get:*:image/jpeg:DLNA.ORG_PN=JPEG_LRG',
					'http-get:*:video/mpeg:DLNA.ORG_PN=AVC_TS_HD_50_AC3_ISO',
					'http-get:*:video/mpeg:DLNA.ORG_PN=AVC_TS_HD_60_AC3_ISO',
					'http-get:*:video/mpeg:DLNA.ORG_PN=AVC_TS_HP_HD_AC3_ISO',
					'http-get:*:video/mpeg:DLNA.ORG_PN=AVC_TS_MP_HD_AAC_MULT5_ISO',
					'http-get:*:video/mpeg:DLNA.ORG_PN=AVC_TS_MP_HD_AC3_ISO',
					'http-get:*:video/mpeg:DLNA.ORG_PN=AVC_TS_MP_HD_MPEG1_L3_ISO',
					'http-get:*:video/mpeg:DLNA.ORG_PN=AVC_TS_MP_SD_AAC_MULT5_ISO',
					'http-get:*:video/mpeg:DLNA.ORG_PN=AVC_TS_MP_SD_AC3_ISO',
					'http-get:*:video/mpeg:DLNA.ORG_PN=AVC_TS_MP_SD_MPEG1_L3_ISO',
					'http-get:*:video/mpeg:DLNA.ORG_PN=MPEG_PS_NTSC',
					'http-get:*:video/mpeg:DLNA.ORG_PN=MPEG_PS_PAL',
					'http-get:*:video/mpeg:DLNA.ORG_PN=MPEG_TS_HD_NA_ISO',
					'http-get:*:video/mpeg:DLNA.ORG_PN=MPEG_TS_SD_NA_ISO',
					'http-get:*:video/mpeg:DLNA.ORG_PN=MPEG_TS_SD_EU_ISO',
					'http-get:*:video/mpeg:DLNA.ORG_PN=MPEG1',
					'http-get:*:video/mp4:DLNA.ORG_PN=AVC_MP4_MP_SD_AAC_MULT5',
					'http-get:*:video/mp4:DLNA.ORG_PN=AVC_MP4_MP_SD_AC3',
					'http-get:*:video/mp4:DLNA.ORG_PN=AVC_MP4_BL_CIF15_AAC_520',
					'http-get:*:video/mp4:DLNA.ORG_PN=AVC_MP4_BL_CIF30_AAC_940',
					'http-get:*:video/mp4:DLNA.ORG_PN=AVC_MP4_BL_L31_HD_AAC',
					'http-get:*:video/mp4:DLNA.ORG_PN=AVC_MP4_BL_L32_HD_AAC',
					'http-get:*:video/mp4:DLNA.ORG_PN=AVC_MP4_BL_L3L_SD_AAC',
					'http-get:*:video/mp4:DLNA.ORG_PN=AVC_MP4_HP_HD_AAC',
					'http-get:*:video/mp4:DLNA.ORG_PN=AVC_MP4_MP_HD_1080i_AAC',
					'http-get:*:video/mp4:DLNA.ORG_PN=AVC_MP4_MP_HD_720p_AAC',
					'http-get:*:video/mp4:DLNA.ORG_PN=MPEG4_P2_MP4_ASP_AAC',
					'http-get:*:video/mp4:DLNA.ORG_PN=MPEG4_P2_MP4_SP_VGA_AAC',
					'http-get:*:video/vnd.dlna.mpeg-tts:DLNA.ORG_PN=AVC_TS_HD_50_AC3',
					'http-get:*:video/vnd.dlna.mpeg-tts:DLNA.ORG_PN=AVC_TS_HD_50_AC3_T',
					'http-get:*:video/vnd.dlna.mpeg-tts:DLNA.ORG_PN=AVC_TS_HD_60_AC3',
					'http-get:*:video/vnd.dlna.mpeg-tts:DLNA.ORG_PN=AVC_TS_HD_60_AC3_T',
					'http-get:*:video/vnd.dlna.mpeg-tts:DLNA.ORG_PN=AVC_TS_HP_HD_AC3_T',
					'http-get:*:video/vnd.dlna.mpeg-tts:DLNA.ORG_PN=AVC_TS_MP_HD_AAC_MULT5',
					'http-get:*:video/vnd.dlna.mpeg-tts:DLNA.ORG_PN=AVC_TS_MP_HD_AAC_MULT5_T',
					'http-get:*:video/vnd.dlna.mpeg-tts:DLNA.ORG_PN=AVC_TS_MP_HD_AC3',
					'http-get:*:video/vnd.dlna.mpeg-tts:DLNA.ORG_PN=AVC_TS_MP_HD_AC3_T',
					'http-get:*:video/vnd.dlna.mpeg-tts:DLNA.ORG_PN=AVC_TS_MP_HD_MPEG1_L3',
					'http-get:*:video/vnd.dlna.mpeg-tts:DLNA.ORG_PN=AVC_TS_MP_HD_MPEG1_L3_T',
					'http-get:*:video/vnd.dlna.mpeg-tts:DLNA.ORG_PN=AVC_TS_MP_SD_AAC_MULT5',
					'http-get:*:video/vnd.dlna.mpeg-tts:DLNA.ORG_PN=AVC_TS_MP_SD_AAC_MULT5_T',
					'http-get:*:video/vnd.dlna.mpeg-tts:DLNA.ORG_PN=AVC_TS_MP_SD_AC3',
					'http-get:*:video/vnd.dlna.mpeg-tts:DLNA.ORG_PN=AVC_TS_MP_SD_AC3_T',
					'http-get:*:video/vnd.dlna.mpeg-tts:DLNA.ORG_PN=AVC_TS_MP_SD_MPEG1_L3',
					'http-get:*:video/vnd.dlna.mpeg-tts:DLNA.ORG_PN=AVC_TS_MP_SD_MPEG1_L3_T',
					'http-get:*:video/vnd.dlna.mpeg-tts:DLNA.ORG_PN=MPEG_TS_HD_NA',
					'http-get:*:video/vnd.dlna.mpeg-tts:DLNA.ORG_PN=MPEG_TS_HD_NA_T',
					'http-get:*:video/vnd.dlna.mpeg-tts:DLNA.ORG_PN=MPEG_TS_SD_EU',
					'http-get:*:video/vnd.dlna.mpeg-tts:DLNA.ORG_PN=MPEG_TS_SD_EU_T',
					'http-get:*:video/vnd.dlna.mpeg-tts:DLNA.ORG_PN=MPEG_TS_SD_NA',
					'http-get:*:video/vnd.dlna.mpeg-tts:DLNA.ORG_PN=MPEG_TS_SD_NA_T',
					'http-get:*:video/x-ms-wmv:DLNA.ORG_PN=WMVSPLL_BASE',
					'http-get:*:video/x-ms-wmv:DLNA.ORG_PN=WMVSPML_BASE',
					'http-get:*:video/x-ms-wmv:DLNA.ORG_PN=WMVSPML_MP3',
					'http-get:*:video/x-ms-wmv:DLNA.ORG_PN=WMVMED_BASE',
					'http-get:*:video/x-ms-wmv:DLNA.ORG_PN=WMVMED_FULL',
					'http-get:*:video/x-ms-wmv:DLNA.ORG_PN=WMVMED_PRO',
					'http-get:*:video/x-ms-wmv:DLNA.ORG_PN=WMVHIGH_FULL',
					'http-get:*:video/x-ms-wmv:DLNA.ORG_PN=WMVHIGH_PRO',
					'http-get:*:video/3gpp:DLNA.ORG_PN=MPEG4_P2_3GPP_SP_L0B_AAC',
					'http-get:*:video/3gpp:DLNA.ORG_PN=MPEG4_P2_3GPP_SP_L0B_AMR',
					'http-get:*:audio/mpeg:DLNA.ORG_PN=MP3',
					'http-get:*:audio/x-ms-wma:DLNA.ORG_PN=WMABASE',
					'http-get:*:audio/x-ms-wma:DLNA.ORG_PN=WMAFULL',
					'http-get:*:audio/x-ms-wma:DLNA.ORG_PN=WMAPRO',
					'http-get:*:audio/x-ms-wma:DLNA.ORG_PN=WMALSL',
					'http-get:*:audio/x-ms-wma:DLNA.ORG_PN=WMALSL_MULT5',
					'http-get:*:audio/mp4:DLNA.ORG_PN=AAC_ISO_320',
					'http-get:*:audio/3gpp:DLNA.ORG_PN=AAC_ISO_320',
					'http-get:*:audio/mp4:DLNA.ORG_PN=AAC_ISO',
					'http-get:*:audio/mp4:DLNA.ORG_PN=AAC_MULT5_ISO',
					'http-get:*:audio/L16;rate=44100;channels=2:DLNA.ORG_PN=LPCM',
					'http-get:*:image/jpeg:*',
					'http-get:*:video/avi:*',
					'http-get:*:video/divx:*',
					'http-get:*:video/x-matroska:*',
					'http-get:*:video/mpeg:*',
					'http-get:*:video/mp4:*',
					'http-get:*:video/x-ms-wmv:*',
					'http-get:*:video/x-msvideo:*',
					'http-get:*:video/x-flv:*',
					'http-get:*:video/x-tivo-mpeg:*',
					'http-get:*:video/quicktime:*',
					'http-get:*:audio/mp4:*',
					'http-get:*:audio/x-wav:*',
					'http-get:*:audio/x-flac:*',
					'http-get:*:application/ogg:*'
				])

			#Samsung TVs are special...
			self.server.content_directory_server.register_vendor_variable(
				'X_FeatureList',
				evented='no',
				data_type='string',
				default_value="")

			self.server.content_directory_server.register_vendor_action(
				'X_GetFeatureList', 'optional',
				(('FeatureList', 'out', 'X_FeatureList'),),
				needs_callback=False)

	def upnp_X_GetFeatureList(self,**kwargs):
		Log.w()
		attrib = {
				"xmlns" : "urn:schemas-upnp-org:av:avs",
				"xmlns:xsi" : "http://www.w3.org/2001/XMLSchema-instance",
				"xsi:schemaLocation" : "urn:schemas-upnp-org:av:avs http://www.upnp.org/schemas/av/avs.xsd"
			}
		features = ET.Element("Features")
		features.attrib.update(attrib)

		attrib = {
				"name" : "samsung.com_BASICVIEW",
				"version" : "1"
			}
		feature = ET.SubElement(features, "Feature")
		feature.attrib.update(attrib)
		#audio/video container id definition
		tag = ET.SubElement(feature, "container")
		tag.attrib.update({ "type": DIDLLite.AudioItem.upnp_class, "id" : str(self._audio.get_id()) })
		tag = ET.SubElement(feature, "container")
		tag.attrib.update({ "type": DIDLLite.VideoItem.upnp_class, "id" : str(self._video.get_id()) })
		return {"FeatureList" : ET.tostring(features, "utf-8")}

def restartMediaServer(name, uuid, **kwargs):
	cp = resourcemanager.getResource("UPnPControlPoint")
	if cp:
		removeUPnPDevice(uuid, cp)
		return cp.registerServer(DreamboxMediaStore, name=name, uuid=uuid, **kwargs)
	return None

