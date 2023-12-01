from twisted.internet import reactor, task
from twisted.internet.defer import Deferred
from twisted.web.resource import Resource
from twisted.web.server import Site

from coherence.extern import louie
from coherence.upnp.core.device import RootDevice
from coherence.upnp.core.ssdp import SSDPServer
from router import ClientRouter, ST, URL
from webrequests import proxy_to, get
from ssdpalt import SSDPServerAlt

import xml.etree.ElementTree as ElementTree
ElementTree.register_namespace('upnp', 'urn:schemas-upnp-org:metadata-1-0/upnp/')
ElementTree.register_namespace('didl', 'urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/')

try:
	import cStringIO as StringIO
except:
	import StringIO

import functools
import json
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import socket
from urllib import quote as urlquote
from urlparse import urljoin

REMOTE_SERVERS = ['http://badasp:8080/devices/']
pollers = {}
ssdp = SSDPServer()   # respond to m-search
ssdpalt = SSDPServerAlt()   # send notifies


def ensure_utf8_bytes(v):
	""" Glibly stolen from the Klein source code """
	if isinstance(v, unicode):
		v = v.encode("utf-8")
	return v
def ensure_utf8(fun):
	@functools.wraps(fun)
	def decorator(*args, **kwargs):
		return ensure_utf8_bytes(fun(*args, **kwargs))
	return decorator

class AltDevice(object):
	def _get_local_ip(self):
		s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		s.connect(('8.8.8.8', 80))
		ip = s.getsockname()[0]
		s.close()
		return ip
	def __init__(self, remote_url):
		resource = UpnpClientResource(remote_url)
		factory = Site(resource)

		self.server = reactor.listenTCP(0, factory)
		host = self.server.getHost()
		proxylocation = 'http://%s:%s/'%(self._get_local_ip(), host.port)
		logger.info("Creating alt device proxy at %s to %s"%(proxylocation, remote_url))
		self.url = proxylocation
	def stop(self):
		self.server.stopListening()

class AltDeviceManager(object):
	def __init__(self):
		self.devices = {}

	def get_device(self, baseurl, uuidport):
		# baseurl is http://remoteserver/devices/
		# uuidport is uuid:74hjasncv:4040
		if not uuidport in self.devices:
			if baseurl[-1] != '/':
				baseurl = baseurl + '/'
			remote_url = baseurl + uuidport
			self.devices[uuidport] = AltDevice(remote_url)
		return self.devices[uuidport]
	def stop(self, uuidport):
		dev = self.devices.get(uuidport, None)
		if dev:
			dev.stop()
			del self.devices[uuidport]
altDeviceManager = AltDeviceManager()

class UpnpClientResource(Resource):
	""" Smarter proxy, with mild routing """
	isLeaf = True

	def __init__(self, remote_url, desc_url=None, device=None):
		if len(remote_url) > 0 and remote_url[-1] != '/':
			remote_url = remote_url + '/'
		self.remote_url = remote_url
		self.router = ClientRouter('/')
		self.router.postprocess(ST.ContentDirectory, URL.controlURL)(self.hack_mediaserver_response)
		if desc_url:
			if len(desc_url) and desc_url[0] == '/':
				desc_url = desc_url[1:]
			self.router.postprocessors[desc_url] = self.hack_description_response
		if device:
			self.set_device(device)

	def set_device(self, device):
		self.device = device
		self.router.add_device(device)

	@ensure_utf8
	def render(self, request):
		return self.router.dispatch_device_request(request, self.get_proxied_url(request.uri))

	def get_proxied_url(self, url):
		# convert a local device url to a proxied url
		if len(url)>1 and url[0] == '/':
			url = url[1:]
		return self.remote_url + url

	def get_altport_url(self, url):
		""" Convert a response's relative url to absolute """
		# url is uuid:74hjasncv:4040/streamid...
		if '/' in url:
			uuidport,rest = url.split('/', 1)
		else:
			uuidport,rest = url, ''

		if uuidport.count(':') != 2:
			# suburl on current device
			return self.localbase + rest
		else:
			# needs an alt device
			base = urljoin(self.remote_url, '..')
			proxy = altDeviceManager.get_device(base, uuidport)
			return proxy.url + rest

	def hack_description_response(self, request, response_data):
		request.setResponseCode(response_data['code'])
		request.responseHeaders = response_data['headers']
		if 'xml' not in response_data['headers'].getRawHeaders('Content-Type', [''])[0]:
			request.responseHeaders.setRawHeaders('Content-Length', [len(response_data['content'])])
			request.write(response_data['content'])
			request.finish()
			return
		request.responseHeaders.removeHeader('Content-Length')
		request.responseHeaders.removeHeader('Content-Encoding')
		# get the device that we're talking to, and its ip
		# load up response
		upnp = 'urn:schemas-upnp-org:device-1-0'
		root = ElementTree.fromstring(response_data['content'])
		for urlbase in root.findall("./{%s}URLBase"%(upnp,)):
			urlbase.text = self.get_altport_url(urlbase.text)
		# write out
		doc = ElementTree.ElementTree(root)
		docout = StringIO.StringIO()
		doc.write(docout, encoding='utf-8', xml_declaration=True)
		docoutstr = docout.getvalue()
		request.responseHeaders.setRawHeaders('Content-Length', [len(docoutstr)])
		request.write(docoutstr)
		request.finish()

	def hack_mediaserver_response(self, request, response_data):
		request.setResponseCode(response_data['code'])
		request.responseHeaders = response_data['headers']
		if 'xml' not in response_data['headers'].getRawHeaders('Content-Type', [''])[0]:
			request.responseHeaders.setRawHeaders('Content-Length', [len(response_data['content'])])
			request.write(response_data['content'])
			request.finish()
			return
		request.responseHeaders.removeHeader('Content-Length')
		request.responseHeaders.removeHeader('Content-Encoding')
		# get the device that we're talking to, and its ip
		# load up response
		didl = 'urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/'
		upnp = 'urn:schemas-upnp-org:metadata-1-0/upnp/'
		root = ElementTree.fromstring(response_data['content'])
		for result in root.iter('Result'):
			resultdoc = ElementTree.fromstring(result.text.encode('utf-8'))
			for uritag in resultdoc.iter('{%s}albumArtURI'%(upnp,)):
				uritag.text = self.get_altport_url(uritag.text).decode('utf-8')
			for uritag in resultdoc.iter('{%s}res'%(didl,)):
				uritag.text = self.get_altport_url(uritag.text).decode('utf-8')
			result.text = ElementTree.tostring(resultdoc, encoding='utf-8').decode('utf-8')
		# write out
		doc = ElementTree.ElementTree(root)
		docout = StringIO.StringIO()
		doc.write(docout, encoding='utf-8', xml_declaration=True)
		docoutstr = docout.getvalue()
		request.responseHeaders.setRawHeaders('Content-Length', [len(docoutstr)])
		request.write(docoutstr)
		request.finish()


class RemoteDevice(object):
	def __init__(self, remote_url, usn, location, st, uuid, server, subdevices):
		# remote url is server/devices/{uuid}
		# location is /desc.xml
		#
		self.uuid = uuid
		self.remote_url = remote_url
		self.location = location

		self.resource = UpnpClientResource(remote_url, desc_url=location, device=None)
		factory = Site(self.resource)
		self.server = reactor.listenTCP(0, factory)
		self.host = self.server.getHost()
		proxylocation = 'http://%s:%s/%s'%(self._get_local_ip(), self.host.port, self.location)
		self.resource.localbase = 'http://%s:%s/'%(self._get_local_ip(), self.host.port)
		logger.info("Creating device proxy for %s at %s"%(remote_url,self.resource.localbase))

		device_infos = {
			'USN': usn,
			'SERVER': server,
			'ST': st,
			'LOCATION': str(proxylocation),
			'MANIFESTATION': 'local',
			'HOST': 'localhost'
		}
		self.load_device_info(device_infos).addCallback(self.advertise)

	def load_device_info(self, device_infos):
		# Return a deferred that signals that the device's info is loaded
		# Bridges the coherence louie event bus to a single deferred
		def receive_callback(device):
			if device == self.device:
				louie.disconnect(receive_callback, 'Coherence.UPnP.RootDevice.detection_completed')
				d.callback(device)
		d = Deferred()
		louie.connect(receive_callback, 'Coherence.UPnP.RootDevice.detection_completed', louie.Any)
		self.device = RootDevice(device_infos)
		return d

	def _get_local_ip(self):
		s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		s.connect(('8.8.8.8', 80))
		ip = s.getsockname()[0]
		s.close()
		return ip

	def advertise(self, device):
		""" After remote parsing is finished, start up listeners """

		logger.info("Creating device proxy at %s to %s"%(device.location, self.remote_url))
		self.resource.set_device(device)
		ssdp.register('local', device.usn, device.st, device.location, device.server, host=self.host.host)
		ssdpalt.register('local', device.usn, device.st, device.location, device.server, host=self.host.host)
		ssdp.register('local', device.get_id(), device.get_id(), device.location, device.server, host=self.host.host)
		ssdpalt.register('local', device.get_id(), device.get_id(), device.location, device.server, host=self.host.host)
		if any(['ContentDirectory' in s.service_type for s in device.get_services()]):
			type = 'urn:schemas-upnp-org:device:MediaServer:1'
			usn = device.get_id() + "::" + type
			ssdp.register('local', usn, type, device.location, device.server, host=self.host.host)
			ssdpalt.register('local', usn, type, device.location, device.server, host=self.host.host)

		for s in device.get_services():
			usn = device.get_id() + "::" + s.service_type
			st = s.service_type
			ssdp.register('local', usn, st, device.location, device.server, host=self.host.host)
			ssdpalt.register('local', usn, st, device.location, device.server, host=self.host.host)
		return device

	def stop(self):
		self.server.stopListening()


class ServerPoller(object):
	def __init__(self, url):
		self.url = url
		self.devices = {}
		self.poller_thread = task.LoopingCall(self.poll)
		self.poller_thread.start(60.0)

	def poll(self):
		# start the connection
		d = get(url, {'Accept':['application/json']})
		d.addCallback(self.on_response)
		d.addErrback(self.on_error)

	def on_response(self, data):
		try:
			obj = json.loads(data['content'])
		except:
			logger.info('Received invalid json data from %s'%(self.url,))
			logger.debug('Received invalid json data from %s: %s'%(self.url, data))
			return

		for device in obj.get('devices', []):
			uuid = device['uuid']
			device_url = urljoin(self.url, urlquote(uuid)) + '/'
			usn = device['usn']
			st = device['st']
			server = device['server']
			location = device['location']  # relative to devices, includes uuid
			location = location.split('/', 1)[-1] # relative to device root
			subdevices = device['subdevices']
			if uuid not in self.devices:
				self.devices[uuid] = RemoteDevice(device_url, usn, location, st, uuid, server, subdevices)

	def on_error(self, err):
		print(err)

for url in REMOTE_SERVERS:
	poller = ServerPoller(url)
	pollers[url] = poller

reactor.run()
