from __future__ import absolute_import
from Components.ActionMap import ActionMap
from Components.config import config, getConfigListEntry
from Components.ConfigList import ConfigListScreen
from Components.Sources.StaticText import StaticText
from Screens.Screen import Screen

from coherence.upnp.core.uuid import UUID

from .DreamboxMediaStore import restartMediaServer
from .UPnPCore import removeUPnPDevice

def getUUID(config_element):
	if config_element.value == "":
		config_element.value = str(UUID())
		config_element.save()
	return config_element.value

class UPnPConfig(ConfigListScreen, Screen):
	skin = """
	<screen name="UPnPConfig" position="center,220" size="720,320" title="UPnP/DLNA Setup">
		<ePixmap pixmap="skin_default/buttons/green.png" position="10,5" size="200,40" alphatest="on" />
		<widget source="key_green" render="Label" position="10,5" size="200,40" zPosition="1" font="Regular;20" halign="center" valign="center" backgroundColor="#1f771f" transparent="1" shadowColor="black" shadowOffset="-2,-2" />
		<eLabel position="10,50" size="700,1" backgroundColor="grey" />
		<widget name="config" position="10,60" size="700,240" enableWrapAround="1" scrollbarMode="showOnDemand" />
	</screen>"""

	def __init__(self, session, args=0):
		Screen.__init__(self, session)
		ConfigListScreen.__init__(self, [], session=session)
		self.setTitle(_("UPnP/DLNA Setup"))

		self["key_green"] = StaticText(_("OK"))
		self["setupActions"] = ActionMap(["SetupActions", "ColorActions"],
		{
			"red": self.close,
			"green": self.close,
			"save": self.close,
			"cancel": self.close,
		}, -2)

		self._addNotifiers()
		self._createSetup()
		self.onClose.append(self._onClose)
		self.onLayoutFinish.append(self.layoutFinished)

	def _addNotifiers(self):
		#server
		config.plugins.mediaserver.enabled.addNotifier(self._enabledChanged, initial_call = False)
		#renderer
		try:
			import Plugins.Extensions.MediaRenderer
			config.plugins.mediarenderer.enabled.addNotifier(self._enabledChanged, initial_call = False)
		except:
			pass

	def _removeNotifiers(self):
		#server
		config.plugins.mediaserver.enabled.removeNotifier(self._enabledChanged)
		if config.plugins.mediaserver.enabled.value:
			restartMediaServer(config.plugins.mediaserver.name.value, getUUID(config.plugins.mediaserver.uuid))
		else:
			removeUPnPDevice( getUUID(config.plugins.mediaserver.uuid) )
		#renderer
		try:
			from Plugins.Extensions.MediaRenderer.plugin import start
			config.plugins.mediarenderer.enabled.removeNotifier(self._enabledChanged)
			if config.plugins.mediarenderer.enabled.value:
				start(0, session=self.session)
			else:
				removeUPnPDevice( getUUID(config.plugins.mediarenderer.uuid) )
		except:
			pass

	def _onClose(self):
		self._removeNotifiers()
		for x in self["config"].list:
			x[1].save()

	def keyLeft(self):
		ConfigListScreen.keyLeft(self)

	def keyRight(self):
		ConfigListScreen.keyRight(self)

	def _createSetup(self):
		l = [getConfigListEntry(_("UPnP/DLNA Mediaserver"), config.plugins.mediaserver.enabled)]
		if config.plugins.mediaserver.enabled.value:
			l.extend( [
				getConfigListEntry(_("Share Audio"), config.plugins.mediaserver.share_audio),
				getConfigListEntry(_("Share Video"), config.plugins.mediaserver.share_video),
				getConfigListEntry(_("Share Live TV/Radio"), config.plugins.mediaserver.share_live),
				getConfigListEntry(_("Server Name"), config.plugins.mediaserver.name),
			])
		try:
			import Plugins.Extensions.MediaRenderer
			l.append(getConfigListEntry(_("UPnP/DLNA MediaRenderer"), config.plugins.mediarenderer.enabled))
			if config.plugins.mediarenderer.enabled.value:
				l.append(getConfigListEntry(_("Renderer Name"), config.plugins.mediarenderer.name))
		except:
			pass
		self["config"].list = l
		self["config"].l.setList(l)

	def _enabledChanged(self, enabled):
		self._createSetup()

	def layoutFinished(self):
		self.setTitle(_("UPnP/DLNA Setup"))

