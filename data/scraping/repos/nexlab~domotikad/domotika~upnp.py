###########################################################################
# Copyright (c) 2011-2014 Unixmedia S.r.l. <info@unixmedia.it>
# Copyright (c) 2011-2014 Franco (nextime) Lanza <franco@unixmedia.it>
#
# Domotika System Controller Daemon "domotikad"  [http://trac.unixmedia.it]
#
# This file is part of domotikad.
#
# domotikad is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##############################################################################

from coherence.base import Coherence
from dmlib.utils import webutils as wu
import logging
from twisted.web import microdom as xml
#import traceback

log = logging.getLogger( 'UPNP' )


class DomotikaUPNP(object):

   def __init__(self, core):
      self.core = core

   def startListen(self):
      log.info("UPNP Service startListen...")
      config = {'logmode':'none', 'interface': self.core.configGet('upnp', 'ethdev')}
      try:
         c = Coherence(config)
         c.connect(self.check_device, 'Coherence.UPnP.Device.detection_completed')
      except:
         log.error("UPNP ERROR")
         #traceback.print_exc(file=open("/tmp/traceback", "w"))
      self.coherence = c
      log.info("UPNP Service started")

   def check_device(self, device):
      log.info("DETECTED DEVICE: "+str(device))
      location=device.get_location()
      wu.getPage(location).addCallback(self.parseXML, device)

   def parseXML(self, res, device):
      descr={'location': 'Unknown',
             'manufacturer': 'Unknown',
             'manufacturerURL': 'Unknown',
             'modelDescription': 'Unknown',
             'modelName': 'Unknown',
             'modelNumber': 'Unknown',
             'deviceType': 'Unknown',
             'host': 'Unknown'}
      try:
         xmlstring=xml.parseXMLString(res)
      except:
         log.info("Cannot parse XML for "+str(device))
         return
      log.debug('RAW DEVICE XML: '+str(res)+' FOR DEVICE: '+str(device))
      xmldev=xml.getElementsByTagName(xmlstring, 'device')[0]
      for k in descr.keys():
         try:
            descr[k] = xml.getElementsByTagName(xmlstring, k)[0].firstChild().toxml()
         except:
            try:
               descr[k] = getattr(device, k, None)
            except:
               descr[k] = 'Unknown'
      log.debug('DEVICE DATA: '+str(descr))
      if 'Network Camera' in descr['deviceType']:
         log.debug("FOUND A CAMERA TO ADD")
         self.core.addMediaSource(descr)
            

      



def startServer(core):
   upnpservice=DomotikaUPNP(core)
   upnpservice.startListen()
   return upnpservice
