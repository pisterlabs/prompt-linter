from __future__ import absolute_import
from Components.config import config
from Components.ResourceManager import resourcemanager
from Plugins.Plugin import PluginDescriptor
from Plugins.SystemPlugins.UPnP.DreamboxMediaStore import restartMediaServer

from .UPnPConfig import UPnPConfig, getUUID

from coherence import log
from coherence.upnp.devices.basics import BasicDeviceMixin, DeviceHttpRoot, RootDeviceXML
from Tools.HardwareInfo import HardwareInfo
from Plugins.SystemPlugins.UPnP.UPnPCore import ManagedControlPoint

class DreamboxUpnpDevice(log.Loggable,BasicDeviceMixin):
	logCategory = 'dreambox'
	device_type = ManagedControlPoint.DEVICE_TYPE_DREAMBOX
	presentationURL = None

	def __init__(self, coherence, backend=None, **kwargs):
		BasicDeviceMixin.__init__(self, coherence, backend, uuid=getUUID(config.plugins.upnp.dreambox_web_uuid), **kwargs)
		self.model_name = HardwareInfo().get_device_name()
		self.device_name = "%s WebInterface" %(self.model_name,)
		self.version = 1
		self._services = []

	def fire(self,backend,**kwargs):
		pass

	def init_complete(self, backend):
		self.web_resource = DeviceHttpRoot(self)
		self.coherence.add_web_resource( str(self.uuid)[5:], self.web_resource)

		transport = "http" if config.plugins.Webinterface.http.enabled.value else "https"
		port = config.plugins.Webinterface.http.port.value
		if not config.plugins.Webinterface.http.enabled.value:
			port = config.plugins.Webinterface.https.port.value
		presentation_url = "%s://%s" %(transport, self.coherence.hostname)
		if (transport == "http" and port != 80) or (transport == "https" and port != 443):
			presentation_url = "%s:%s" %(presentation_url, port)

		self.web_resource.putChild('description-%d.xml' % self.version,
								RootDeviceXML( self.coherence.hostname,
								str(self.uuid),
								self.coherence.urlbase,
								device_type=self.device_type,
								device_uri_base=ManagedControlPoint.URI_BASE_DREAMBOX,
								version=self.version,
								friendly_name=self.device_name,
								services=[],
								devices=[],
								icons=self.icons,
								manufacturer='dreambox',
								manufacturer_url='http://www.dreambox.de',
								model_description=self.model_name,
								model_name=self.model_name,
								model_number=self.device_name,
								model_url='http://www.dreambox.de',
								presentation_url=presentation_url))

		self.register()
		self.warning("%s %s (%s) activated with id %s" % (self.device_type, self.model_name, self.backend, str(self.uuid)[5:]))

def upnp_start(reason, **kwargs):
	if reason == 0:
		try:
			from Plugins.Extensions.WebInterface import webif
			if config.plugins.Webinterface.enabled.value and (config.plugins.Webinterface.http.enabled.value or config.plugins.Webinterface.https.enabled.value):
				cp = resourcemanager.getResource("UPnPControlPoint")
				if cp:
					cp.registerDevice(DreamboxUpnpDevice(cp.coherence))
		except:
			pass
		if config.plugins.mediaserver.enabled.value:
			restartMediaServer(
					config.plugins.mediaserver.name.value,
					getUUID(config.plugins.mediaserver.uuid),
					manufacturer='dreambox',
					manufacturer_url='http://www.dreambox.de',
					model_description='Dreambox MediaServer',
					model_name=config.plugins.mediaserver.name.value,
					model_number=config.plugins.mediaserver.name.value,
					model_url='http://www.dreambox.de'
				)

def session_start(reason, session=None, **kwargs):
	if reason == 0 and session != None:
		cp = resourcemanager.getResource("UPnPControlPoint")
		if cp:
			cp.setSession(session)

def upnp_setup(session, **kwargs):
	session.open(UPnPConfig)

def upnp_menu(menuid, **kwargs):
	if menuid == "network":
		return [(_("UPnP/DLNA"), upnp_setup, "upnp_setup", None)]
	else:
		return []

def Plugins(**kwargs):
	return [PluginDescriptor(where=PluginDescriptor.WHERE_SESSIONSTART, fnc=session_start),
			PluginDescriptor(where=PluginDescriptor.WHERE_UPNP, fnc=upnp_start),
			PluginDescriptor(name=_("UPnP/DLNA Setup"), description=_("Setup UPnP/DLNA Services"), where = PluginDescriptor.WHERE_MENU, needsRestart = True, fnc=upnp_menu)
		]
