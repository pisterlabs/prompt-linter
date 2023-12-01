#!/usr/bin/env python

import logging
from twisted.internet import reactor
from coherence.upnp.core import DIDLLite
from ssdpalt import SSDPServerAlt
from coherence.upnp.core.ssdp import SSDPServer
from coherence.upnp.core.msearch import MSearch
from coherence.upnp.core.device import Device, RootDevice
from coherence.extern import louie

logger = logging.getLogger(__name__)

class DeviceManager(object):
	def __init__(self):
		self.ssdp = SSDPServer()		# listen for things destined to port 1900
		self.ssdpalt = SSDPServerAlt()
		self.msearch = MSearch(self.ssdpalt, test=False)	# use correct source port
		self.devices = []
		self.orphans = {}
		self.listeners = {'added':[], 'deleted':[]}
		louie.connect(self.ssdp_detected, 'Coherence.UPnP.SSDP.new_device', louie.Any)
		louie.connect(self.ssdp_deleted, 'Coherence.UPnP.SSDP.removed_device', louie.Any)
		louie.connect(self.device_found, 'Coherence.UPnP.RootDevice.detection_completed', louie.Any)
		self.msearch.double_discover()

	def register(self, event, callback):
		self.listeners[event].append(callback)

	def unregister(self, event, callback):
		if callback in self.listeners[event]:
			self.listeners[event].remove(callback)

	def _send_event(self, event, args=(), kwargs={}):
		for callback in self.listeners[event]:
			callback(*args, **kwargs)

	def _get_device_by_id(self, id):
		found = None
		for device in self.devices:
			this_id = device.get_id()
			if this_id[:5] != 'uuid:':
				this_id = this_id[5:]
			if this_id == id:
				found = device
				break
		return found

	def _get_device_by_usn(self, usn):
		found = None
		for device in self.devices:
			if device.get_usn() == usn:
				found = device
				break
		return found

	def ssdp_detected(self, device_type, infos, *args, **kwargs):
		logger.debug("SSDP announced: %s"%(infos,))
		if infos['ST'] == 'upnp:rootdevice':
			root = RootDevice(infos)	# kicks off loading of the device info
			                        	# which will call device_found callback
			root_id = infos['USN']
			for orphan in self.orphans.get(root_id, []):
				orphan.parent = root
		else:
			logger.debug("Find subdevice %s"%(infos,))
			root_id = infos['USN'][:-len(infos['ST']) - 2]
			root = self._get_device_by_id(root_id)
			if root:
				device = Device(infos, root)
				root.add_device(device)
				self.device_found(device)
			else:
				device = Device(infos)
				wants_parent = self.orphans.get(root_id, [])
				wants_parent.append(device)
				self.orphans[root_id] = wants_parent
	def ssdp_deleted(self, device_type, infos, *args, **kwargs):
		device = self._get_device_by_usn(infos['USN'])
		if device:
			louie.send('Coherence.UPnP.Device.removed', None, usn=infos['USN'])
			self._send_event('deleted', args=(device,))
			self.devices.remove(device)
			device.remove()
			if infos['ST'] == 'upnp:rootdevice':
				louie.send('Coherence.UPnP.RootDevice.removed', None, usn=infos['USN'])

	def device_found(self, device):
		logger.debug("UPNP Device discovered: %s"%(device,))
		self.devices.append(device)
		self._send_event('added', args=(device,))
	
	def browse_callback(self, result):
		results = DIDLLite.DIDLElement.fromString(result['Result']).getItems()
		print([result.title for result in results])
	def browse_error(self, error):
		print(error.getTraceback())

