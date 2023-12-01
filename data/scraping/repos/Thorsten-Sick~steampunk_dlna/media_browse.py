# This tool browses through the music collection available by UPNP/DLNA and creates
# a json document with all songs and their meta data

# Original: http://coherence-project.org/browser/trunk/Coherence/misc/media_server_observer.py
# Which is also an example on the homepage
# Additional inspiration:
# http://coherence-project.org/browser/trunk/Coherence/misc/Rhythmbox-Plugin/upnp_coherence/UpnpSource.py

from twisted.internet import reactor

from coherence.base import Coherence
from coherence.upnp.devices.control_point import ControlPoint
from coherence.upnp.core import DIDLLite

import json

class Music_DB():
    def __init__(self):
        self.songs = {}
        self.filename = "/home/thorsten/removeme/music_data.json"

    def __str__(self):
        res = str(self.albums)

    def add_song(self, item, nurl, duration):
        if nurl in self.songs:
            return


        self.songs[nurl] = {"title": item.title,
                            "album": item.album,
                            "artist": item.artist,
                            "albumArtURI": item.albumArtURI,
                            "genre": item.genre,
                            "date": item.date,
                            "description": item.description,
                            "id": item.id,
                            "originalTrackNumber": item.originalTrackNumber,
                            "duration": duration
                            }
        with open(self.filename, "wt") as fh:
            json.dump(self.songs, fh, indent=4)



class UPNP_Browser():
    def __init__(self):
        self.browse_count = 0
        self.music = Music_DB()
        self.__client = None
        self.control_point = ControlPoint(Coherence({'logmode':'warning'}),
            auto_client=['MediaServer'])
        self.control_point.connect(self.media_server_found,
                'Coherence.UPnP.ControlPoint.MediaServer.detected')
        self.control_point.connect(self.media_server_removed,
                'Coherence.UPnP.ControlPoint.MediaServer.removed')


        # now we should also try to discover the ones that are already there:
        for device in self.control_point.coherence.devices:
            print device

    # called for each media server found
    def media_server_found(self, client, udn):
        print "media_server_found", client
        print "media_server_found", client.device.get_friendly_name()

        self.__client = client
        self.load_children(0)

    # sadly they sometimes get removed as well :(
    def media_server_removed(self, udn):
        print "media_server_removed", udn

    # browse callback
    def process_media_server_browse(self, result, client):
        print "browsing root of", client.device.get_friendly_name()
        print "result contains %d out of %d total matches" % \
                (int(result['NumberReturned']), int(result['TotalMatches']))

        elt = DIDLLite.DIDLElement.fromString(result['Result'])
        for item in elt.getItems():
            if item.upnp_class.startswith("object.container"):
                self.load_children(item.id)

            if item.upnp_class.startswith("object.item"):
                url = None
                duration = None
                size = None

                for res in item.res:
                    remote_protocol,remote_network,remote_content_format,remote_flags = res.protocolInfo.split(':')
                    if remote_protocol == 'http-get':
                        url = res.data
                        duration = res.duration
                        size = res.size
                        break

                self.music.add_song(item, url, duration)

    def load_children(self, id):
        self.browse_count += 1
        d = self.__client.content_directory.browse(id, browse_flag='BrowseDirectChildren', process_result=False, backward_compatibility=False)
        d.addCallback(self.process_media_server_browse, self.__client)
        #d.addErrback(self.err_back)

def start():
    bowser = UPNP_Browser()

if __name__ == "__main__":
    reactor.callWhenRunning(start)
    reactor.run()