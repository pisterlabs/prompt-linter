import time
import os

from twisted.internet import reactor
from coherence.base import Coherence
from coherence.upnp.devices.control_point import ControlPoint
from coherence.upnp.core import DIDLLite
from coherence.extern.config import Config

from EventHandler import EventHandler, Event
from XmlData import XmlData, Attribute
from Config import config
from BackendDevice import Album, Song, Movie, Picture, BackendDevice

import XmlGenerator

class BackendUpnp:
    def __init__(self, handler, configfile, devicename, devicetype):
        self._devices = []
        self._handler = handler
        self._configfile = configfile
        self._localdevicename = devicename
	self._localdevicetype = devicetype

    '''
    -------------------
     GETTERS & SETTERS 
    -------------------
    '''

    def get_devices(self):
        return self._devices


    def get_device_by_udn(self, udn):
        for device in self._devices:
            if device.get_uid() == udn:
                return device 
            else:
                pass


    def get_device_by_name(self, name):
        for device in self._devices:
            client = device.get_client()
            if client.device.get_friendly_name() == name:
                return device
            else:
                pass

    def create_local_device(self, device):
	pass

    '''
     is called when controlpoint detects a mediaserver
    '''

    def mediaserver_detected(self,client,udn):
        if client is not None:
            print "device detected: ", client.device.get_friendly_name()
            print "local device name: ", self._localdevicename

            udn = self._parseUUID(udn)
            
            alreadyinlist = 0

            # add into an empty list
            # quick & dirty workaround: coherence & controlpoint detect things multiple times..
            for d in self._devices:
                if d.get_uid() == udn:
                    alreadyinlist = 1
            
            if len(self._devices) == 0 or alreadyinlist == 0:
#                device = BackendDevice(client, udn, str("phone"))
                device = BackendDevice(client, udn, config.device_type)
		
		# check if the device is the local one
		if client.device.get_friendly_name() == self._localdevicename:
		    self._localdevice = device

                self._devices.append(device)

            print "count: ", len(self._devices)
            self._createXML() 
            self.browse_mediaserver(udn)
	    self._createXMLfilelist()

        else:
            self.close()



    '''
     is called when controlpoint detects a mediaserver removal
    '''

    def mediaserver_removed(self,udn):
        if len(self._devices) >= 1:
            device = self.get_device_by_udn(udn)
            if device is not None:
                print "device removed: ", device.get_client().device.get_friendly_name()
                self._devices.remove(device)
                print "count ", len(self._devices)
                
                self._createXML()



    '''
     use this function to start browsing.
    '''

    def browse_mediaserver(self, udn):
        device = self.get_device_by_udn(udn)
        client = device.get_client()

        d = client.content_directory.browse(0,
                                            browse_flag='BrowseDirectChildren',
                                            process_result=False,
                                            backward_compatibility=False)
        d.addCallback(self._process_mediaserver_browse, client)



    '''
     check content folders, forward to correct callback
    '''

    def _process_mediaserver_browse(self, result, client):
        elt = DIDLLite.DIDLElement.fromString(result['Result'])
        for item in elt.getItems():
            
            if hasattr(item, 'childCount') and item.childCount >0:
                if item.title == "Music":
                    d = client.content_directory.browse(item.id,
                                                        browse_flag='BrowseDirectChildren',
                                                        process_result=False,
                                                        backward_compatibility=False)
                    d.addCallback(self._process_mediaserver_browse_music, client, 0)

                elif item.title == "Pictures":
                    d = client.content_directory.browse(item.id,
                                                        browse_flag='BrowseDirectChildren',
                                                        process_result=False,
                                                        backward_compatibility=False)
                    d.addCallback(self._process_mediaserver_browse_pictures, client)
                    
                elif item.title == "Videos":
                    d = client.content_directory.browse(item.id,
                                                        browse_flag='BrowseDirectChildren',
                                                        process_result=False,
                                                        backward_compatibility=False)
                    d.addCallback(self._process_mediaserver_browse_video, client, 0)
                    


    '''
     process music directory browsing
    '''

    def _process_mediaserver_browse_music(self, result, client, recurselevel):
        elt = DIDLLite.DIDLElement.fromString(result['Result'])
        for item in elt.getItems():
            
            if hasattr(item, 'childCount') and item.childCount >0:
                d = client.content_directory.browse(item.id,
                                                    browse_flag='BrowseDirectChildren',
                                                    process_result=False,
                                                    backward_compatibility=False)
                d.addCallback(self._process_mediaserver_browse_music, client, recurselevel+1)

            if item.upnp_class.startswith("object.container") and recurselevel == 1:
                # album
                URL = client.device.get_host() + ":30020/" + self._parseUUID(client.device.get_id()) + "/"
                album = Album(item.id, item.title, "", URL+item.id)
                self._localdevice.add_album(album)

            if item.upnp_class.startswith("object.item") and recurselevel == 2:
                # song
                album = self._localdevice.find_album( item.parentID )
                if album is not None and item.title.endswith(".mp3"):
                    album.add_song( Song(item.id, item.title) )
                elif album is not None and item.title.startswith("cover") and item.title.endswith((".jpg",".png",".gif")):
                    album.set_cover(item.title)


            # user error...
            if recurselevel > 2:
                return

            self._createXMLfilelist()


    '''
     process picture directory browsing
    '''

    def _process_mediaserver_browse_pictures(self, result, client):
        elt = DIDLLite.DIDLElement.fromString(result['Result'])
        for item in elt.getItems():
            if item.upnp_class.startswith("object.item"):
                # picture
                picture = self._localdevice.find_picture( item.id )
                URL = client.device.get_host() + ":30020/" + self._parseUUID(client.device.get_id()) + "/"
                if picture is None and item.title.endswith((".jpg",".png",".gif")):
                    self._localdevice.add_picture( Picture(item.id, URL+item.title, URL+item.title) )

            self._createXMLfilelist()

    '''
     process video directory browsing
    '''

    def _process_mediaserver_browse_video(self, result, client, recurselevel):
        elt = DIDLLite.DIDLElement.fromString(result['Result'])
        for item in elt.getItems():
            
            if hasattr(item, 'childCount') and item.childCount >0:
                d = client.content_directory.browse(item.id,
                                                    browse_flag='BrowseDirectChildren',
                                                    process_result=False,
                                                    backward_compatibility=False)
                d.addCallback(self._process_mediaserver_browse_video, client, recurselevel+1)

            if item.upnp_class.startswith("object.container") and recurselevel == 0:
                # folder
                movie = Movie(item.id, "", item.title, "")
                self._localdevice.add_movie(movie)

            if item.upnp_class.startswith("object.item") and recurselevel == 1:
                # file
                movie = self._localdevice.find_movie( item.parentID )
                if movie is not None and item.title.endswith((".avi",".mp4",".mpeg",".mov",".wmv",".flv")):
                    URL = client.device.get_host() + ":30020/" + self._parseUUID(client.device.get_id()) + "/"
                    movie.set_uri(URL+item.title)
                elif movie is not None and item.title.startswith("cover") and item.title.endswith((".jpg",".png",".gif")):
                    movie.set_cover(item.title)


            # user error...
            if recurselevel > 1:
                return

            self._createXMLfilelist()




    '''
     create an XML file containing the device list
    '''

    def _createXML(self):
        path = config.temp_folder
        deviceroot = XmlData("devices")
        devicelist = "devicelist.xml"

        if not os.path.exists(path):
            os.makedirs(path)
        
        # iterate thru all devices
        for device in self._devices:
            client = device.get_client()

            child = XmlData("device")
            child.addAttribute( Attribute("type", device.get_type() ) )
            child.addAttribute( Attribute("id", device.get_uid()) )
            child.addAttribute( Attribute("name", client.device.get_friendly_name()) )
            
            deviceroot.addChild(child)
        


        # write the list to xml
        XmlGenerator.xmlToFile(deviceroot, path+devicelist)

        # notify the UI that our xml data has changed
        self._handler.handleEvent( 
            Event( "ReturnUiData", "UpnpBackend", "DeviceListChanged", path+devicelist ) )



    '''
     create local filelist
    '''
    def _createXMLfilelist(self):	
#        path = "/tmp/mymeego/data/"
        path = config.temp_folder
	fileroot = XmlData("transferfiles")
	filelist = "filelist.xml"

	# write local files to xml
	device = self._localdevice

        # albums
    	albumroot = XmlData("albums")

    	for album in device.get_albums():
	    child = XmlData("album")
	    child.addAttribute( Attribute("id", album.get_id() ))
	    child.addAttribute( Attribute("title", album.get_name() ))
	    child.addAttribute( Attribute("cover", album.get_cover() ))
	    child.addAttribute( Attribute("URL", album.get_uri() ))

	    songroot = XmlData("songs")

	    for song in album._songs:
	        songchild = XmlData("song")
	        songchild.addAttribute( Attribute("number", song.get_number() ))
	        songchild.addAttribute( Attribute("filename", song.get_filename() ))
		
	        songroot.addChild(songchild)

            child.addChild(songroot)
	    albumroot.addChild(child)
	
	fileroot.addChild(albumroot)

        # pictures
        pictureroot = XmlData("pictures")

        for picture in device.get_pictures():
            child = XmlData("picture")
            child.addAttribute( Attribute("id", picture.get_id()))
            child.addAttribute( Attribute("thumbnail", picture.get_thumbnail()))
            child.addAttribute( Attribute("URL", picture.get_uri()))

            pictureroot.addChild(child)

        fileroot.addChild(pictureroot)

        # movies 
        movieroot = XmlData("movies")

        for movie in device.get_movies():
            child = XmlData("movie")
            child.addAttribute( Attribute("id", movie.get_id()))
            child.addAttribute( Attribute("cover", movie.get_cover()))
            child.addAttribute( Attribute("name", movie.get_name()))
            child.addAttribute( Attribute("URL", movie.get_uri()))

            movieroot.addChild(child)

        fileroot.addChild(movieroot)

	XmlGenerator.xmlToFile(fileroot, path+filelist)



    '''
     sanitize UUID, upnp adds "uuid:" to the beginning by default, which messes things up
    '''

    def _parseUUID(self, udn):
        parsedID = udn.partition("uuid:")
        return parsedID[2]
    

    '''
     remove the XML files
    '''
    def _removeXML(self):
        devicelist = "devicelist.xml"
        filelist = "filelist.xml"
        path = config.temp_folder

        if os.path.exists(path+devicelist):
            os.remove(path+devicelist)

        if os.path.exists(path+filelist):
            os.remove(path+filelist)


    '''
    --------------------------------------------------
     REACTOR operations below.
     contains start script, main loop and end command.
    --------------------------------------------------
    '''

    def run(self):
        reactor.callWhenRunning(self.start)
        reactor.run()


    def start(self):
        print "Upnp backend starting"
        # clear device list
        self._removeXML() 

        # config Coherence
        conf = Config(self._configfile)
        c = Coherence({'logmode':'warning'})
        c.setup(conf)        
        #add share folders
        con = ""
        for f in config.share_folders:
            if len(con) > 0:
                con +=","
            con += f
        c.add_plugin('FSStore',name=config.device_name,content=con)

        # start a ControlPoint to start detecting upnp activity
        cp = ControlPoint(c, auto_client=['MediaServer'])
        cp.connect(self.mediaserver_detected, 'Coherence.UPnP.ControlPoint.MediaServer.detected')
        cp.connect(self.mediaserver_removed, 'Coherence.UPnP.ControlPoint.MediaServer.removed')
	
	#self._localdevice = cp.get_device_by_name(self._localdevicename)
	#print "SPRURPDDSOPF            ",self._localdevice.client.get_friendly_name()
        
        print "Upnp loop succesfully started"
        self._handler.start()
        print "DBus loop succesfully started"


    def close(self):
        print "upnp close()"
        if reactor.running == True:
            reactor.stop()
            self._removeXML()
            self._handler.close()

