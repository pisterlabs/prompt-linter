from Components.ResourceManager import resourcemanager

from coherence.upnp.core import DIDLLite
from UPnPCore import Statics, Item

class UPnPBrowser(object):
	def __init__(self):
		self.controlPoint = resourcemanager.getResource("UPnPControlPoint")
		assert self.controlPoint is not None

		self._currentClient = None
		self._idHistory = []
		self._currentID = -1
		self._requestedID = -1

		self.onListReady = []
		self.onBrowseError = []
		self.list = []

		self.onMediaServerDetected = []
		self.onMediaServerRemoved  = []

		self.currentPath = ""

		self.controlPoint.onMediaServerDetected.append(self._onMediaServerDetected)
		self.controlPoint.onMediaServerRemoved.append(self._onMediaServerRemoved)

	def __del__(self):
		self.controlPoint.onMediaServerDetected.remove(self._onMediaServerDetected)
		self.controlPoint.onMediaServerRemoved.remove(self._onMediaServerRemoved)

	def _setClient(self, client):
		self._currentClient = client
		self._currentID = -1
		self._idHistory = []

	'''
	Returns a list of server-udns (ids), use Item.getServerName(client) to get the actual name of the server for any known udn
	'''
	def getServerList(self):
		return self.controlPoint.getServerList()

	def getList(self):
		return self.list

	def getItemType(self, item):
		return Item.getItemType(item)

	def isServer(self, item):
		return Item.isServer(item)

	'''
	Returns the title of the current item
	'''
	def getItemTitle(self, item):
		return Item.getItemTitle(item)

	'''
	returns the number of children for container items
	returns -1 for non-container items
	'''
	def getItemChildCount(self, item):
		return Item.getItemChildCount(item)

	'''
	Currently always returns a dict with the first url and meta-type, which is usually the original/non-transcoded source
	Raises an IllegalInstanceException if you pass in a container-item
	'''
	def getItemUriMeta(self, item):
		return Item.getItemUriMeta(item)

	def getItemId(self, item):
		return Item.getItemId(item)

	def getItemMetadata(self, item):
		return Item.getItemMetadata(item)

	def getAudioAlbums(self, start_index=0, item_count=500, server=None):
		'''
		search requestes and sort criteria for audio albums taken from
		minidlna in debug mode based on some popular device's resquests:

		* SearchCriteria: (upnp:class = "object.container.album.musicAlbum")
		* Filter: dc:title,upnp:artist
		* SortCriteria: +dc:title
		'''
		criteria = '(upnp:class = "object.container.album.musicAlbum")'
		self.search(self, 0, criteria, start_index=start_index, item_count=item_count, item=server)

	def getAudioArtists(self, start_index=0, item_count=500, server=None):
		'''
		search requestes and sort criteria for audio artists taken from
		minidlna in debug mode based on some popular device's resquests:

		* SearchCriteria: (upnp:class ="object.container.person.musicArtist")
		* Filter: dc:title
		* SortCriteria: +dc:title
		'''
		criteria = '(upnp:class ="object.container.person.musicArtist")'
		self.search(self, 0, criteria, start_index=start_index, item_count=item_count, item=server)

	def getAudioGenres(self, start_index=0, item_count=500, server=None):
		'''
		search requestes and sort criteria for audio genres taken from
		minidlna in debug mode based on some popular device's resquests:

		* SearchCriteria: (upnp:class = "object.container.genre.musicGenre")
		* Filter: dc:title
		* SortCriteria: +dc:title
		'''
		criteria = '(upnp:class = "object.container.genre.musicGenre")'
		self.search(self, 0, criteria, start_index=start_index, item_count=item_count, item=server)

	def getAudioTitles(self, start_index=0, item_count=500, server=None):
		'''
		search requestes and sort criteria for audio titles taken from
		minidlna in debug mode based on some popular device's resquests:

		* SearchCriteria: (upnp:class derivedfrom "object.item.audioItem")
		* Filter: dc:title,res,res@duration,res@sampleFrequency,res@bitsPerSample,res@bitrate,res@nrAudioChannels,upnp:artist,upnp:artist@role,upnp:genre,upnp:album
		* SortCriteria: +dc:title
		'''
		criteria = '(upnp:class derivedfrom "object.item.audioItem")'
		self.search(self, 0, criteria, start_index=start_index, item_count=item_count, item=server)

	def getPlaylists(self, start_index=0, item_count=500, server=None):
		'''
		search requestes and sort criteria for audio albums taken from
		minidlna in debug mode based on some popular device's resquests:

		* SearchCriteria: (upnp:class = "object.container.playlistContainer")
		* Filter: dc:title
		* SortCriteria: +dc:title
		'''
		criteria = '(upnp:class = "object.container.playlistContainer")'
		self.search(self, 0, criteria, start_index=start_index, item_count=item_count, item=server)

	def canAscend(self):
		if self._currentID >= Statics.CONTAINER_ID_ROOT:
			return True
		return False

	def ascend(self):
		print "[UPnPBrowser].ascend currentID='%s'" %(self._currentID)
		if len(self._idHistory) > 0:
			self._requestedID = self._idHistory[-1]
			self.currentPath = self.currentPath[0:self.currentPath.rfind(' > ')]
			self.browse(container_id=self._requestedID)
			return True
		elif self._currentID == 0:
			self._currentID = Statics.CONTAINER_ID_SERVERLIST
			self.currentPath = _("Servers")
			self._onListReady(self.controlPoint.getServerList())
			return True
		return False

	def canDescend(self, item):
		return self._isContainer(item)

	def descend(self, item = None):
		type = self.getItemType(item)
		title = self.getItemTitle(item)
		if type == Statics.ITEM_TYPE_SERVER:
			self.browse(container_id=Statics.CONTAINER_ID_ROOT, item=item)
			self.currentPath = title
			return True
		elif type == Statics.ITEM_TYPE_CONTAINER:
			self.currentPath = "%s > %s" %(self.currentPath, title)
			self.browse(container_id=item.id)
			return True

		return False

	def refresh(self):
		self.browse(self._currentID)

	def browse(self, container_id=Statics.CONTAINER_ID_SERVERLIST, sort_criteria="", start_index=0, item_count=500, item=None):
		if item is None and container_id == Statics.CONTAINER_ID_SERVERLIST:
			self.currentPath = _("Servers")
			self._onListReady(self.getServerList())
			self._currentID = Statics.CONTAINER_ID_SERVERLIST
			return

		if self.isServer(item):
			self._setClient(item)

		if container_id == Statics.CONTAINER_ID_SERVERLIST:
			Statics.CONTAINER_ID_ROOT

		if self._currentClient == None:
			raise UnboundLocalError("UPnPBrowser.browse called but no valid client assigned")

		print "[UPnPBrowser].browse: %s#%s" %(self._currentClient.device.get_friendly_name(), container_id)
		self._requestedID = container_id

		d = self._currentClient.content_directory.browse(
				container_id,
				browse_flag='BrowseDirectChildren',
				filter='*',
				sort_criteria=sort_criteria,
				process_result=False,
				backward_compatibility=False,
				starting_index=start_index,
				requested_count=item_count)

		#we pass the client to fix any kind of very odd mixup on concurrent requests (which should actually be avoided in the first time
		d.addCallback(self._onBrowseResult, self._currentClient)
		d.addErrback(self._onBrowseError)

	'''
	Coherence currently doesn't support sort criterias in searches, should be fixed in coherence!
	'''
	def search(self, container_id, criteria, sort_criteria="+dc:title", start_index=0, item_count=500, item=None):
		if self.isServer(item):
			self._setClient(item)

		if container_id is None:
			container_id = Statics.CONTAINER_ID_ROOT

		if self._currentClient == None:
			raise UnboundLocalError("UPnPBrowser.search called but no valid item assigned")

		print "[UPnPBrowser].search: '%s#%s" %(self._currentClient.device.get_friendly_name(), container_id)
		self._requestedID = container_id

		d = self._currentClient.content_directory.search(
				container_id,
				criteria,
				start_index,
				item_count,
			)
		#we pass the client to fix any kind of very odd mixup on concurrent requests (which should actually be avoided in the first time
		d.addCallback(self._onBrowseResult, self._currentClient)
		d.addErrback(self._onBrowseError)

	def _onBrowseResult(self, result, client):
		if self._currentClient != client:
			print "WARNING! - [UPnPBrowser]._onBrowseResult - Concurrent Requests detected, please fix your client!"
			self._setClient(client)

		res = DIDLLite.DIDLElement.fromString(result['Result'].encode( "utf-8" ))

		ascended = False
		if len(self._idHistory) > 0:
			#Ascended, remove the item from the history
			if self._requestedID == self._idHistory[-1]:
				self._idHistory.pop()
				ascended = True
		#Descended, add the id of the last item to the history
		if self._currentID >= 0 and not ascended:
			self._idHistory.append(self._currentID)

		self._currentID = self._requestedID

		list = []
		for item in res.getItems():
			list.append(item)
		self._onListReady(list)

	def _onBrowseError(self, err):
		print "[UPnPBrowser]._onBrowseError"
		err.printTraceback()
		self._currentID = self._requestedID
		for fnc in self.onBrowseError:
			fnc(err)

	def _onListReady(self, list):
		self.list = list
		for fnc in self.onListReady:
			fnc(list)

	def _isContainer(self, item):
		return Item.isContainer(item)

	def _onMediaServerDetected(self, udn, client):
		print "[UPnPBrowser]._onMediaServerDetected %s" %udn
		for fnc in self.onMediaServerDetected:
			fnc(udn, client)

	def _onMediaServerRemoved(self, udn):
		print "[UPnPBrowser]._onMediaServerRemoved %s" %udn
		for fnc in self.onMediaServerRemoved:
			fnc(udn)
