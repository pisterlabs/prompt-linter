from twisted.internet import reactor, task
from coherence.base import Coherence, Plugins
from coherence.backend import BackendItem, BackendStore
from coherence.backends.mediadb_storage import MediaStore, Track, KNOWN_AUDIO_TYPES
from coherence.upnp.core.DIDLLite import classChooser, Container
from twisted.python.filepath import FilePath

import random
import os.path
import argparse


class ShortListItem(BackendItem):
    logCategory = "shortlist_item"

    def __init__(
        self,
        object_id,
        parent,
        path,
        mimetype,
        urlbase,
        UPnPClass,
        update=False,
        store=None,
    ):
        BackendItem.__init__(self)
        self.id = object_id

        if mimetype == "root":
            self.update_id = 0
        if mimetype == "item" and path is None:
            path = os.path.join(parent.get_realpath(), str(self.id))
        self.location = path
        self.debug("location %s", self.location)
        self.mimetype = mimetype
        if urlbase[-1] != "/":
            urlbase += "/"
        self.url = urlbase + str(self.id)
        if parent is None:
            parent_id = -1
        else:
            parent_id = parent.get_id()
        self.item = UPnPClass(object_id, parent_id, self.get_name())
        if isinstance(self.item, Container):
            self.item.childCount = 0
        self.children = []
        self.store = store

    def get_id(self):
        return self.id

    def add_child(self, child, update=False):
        self.children.append(child)
        if isinstance(self.item, Container):
            self.item.childCount += 1
        if update:
            self.update_id += 1

    def remove_child(self, child, update=True):
        try:
            self.children.remove(child)
            self.store.remove_item(child)
        except ValueError:
            self.warn(
                "Child item %s was already missing", child
            )  # Generally should exist, but sometimes it doesn't. Shouldn't crash
        if update:
            self.update_id += 1

    def get_name(self):
        if hasattr(self, "display"):
            return self.display
        if isinstance(self.location, FilePath):
            return self.location.basename()
        else:
            return self.location

    def get_children(self, start=0, request_count=0):
        try:
            self.debug("get_children %d %d", start, request_count)
            if request_count == 0:
                return self.children[start:]
            else:
                return self.children[start:request_count]
        except Exception as e:
            self.error(e)
            raise

    def get_child_count(self):
        self.debug("get_child_count")
        return len(self.children)

    def __repr__(self):
        return (
            "id: "
            + str(self.id)
            + " @ "
            + str(self.get_name().encode("ascii", "xmlcharrefreplace"))
            if self.get_name() is not None
            else ""
        )


class ShortListStore(BackendStore):
    logCategory = "shortlist_store"

    implements = ["MediaServer"]

    description = """
    Subset playlist backend based on a mediadb
    backend to workaround track limits
    """

    options = [
        {
            "option": "name",
            "type": "string",
            "default": "ShortlistStore",
            "help": "the name under this MediaServer "
            "shall show up with on other UPnP clients",
        },
        {"option": "medialocation", "type": "string", "help": "path to media"},
        {
            "option": "mediadb",
            "type": "string",
            "help": "path to media database (will be created if doesn't exist)",
        },
        {
            "option": "trackcount",
            "type": "integer",
            "help": "tracks in the playlist",
            "default": 50,
        },
        {
            "option": "updateFrequency",
            "type": "integer",
            "help": "track update frequency in seconds",
            "default": 300,
        },
    ]

    def __init__(
        self,
        server,
        name="ShortlistStore",
        trackcount=50,
        updateFrequency=300,
        **kwargs
    ):
        BackendStore.__init__(self, server, **kwargs)
        self.name = name
        self.next_id = 1000
        self.store = {}
        self.trackcount = trackcount
        self.updateFrequency = updateFrequency
        UPnPClass = classChooser("root")
        id = str(self.getnextID())
        self.root = ShortListItem(
            id, None, "media", "root", self.urlbase, UPnPClass, update=True, store=self
        )
        self.add_store_item(id, self.root)

        self.source_backend = MediaStore(server, **kwargs)

        self.wmc_mapping.update({"14": "0", "15": "0", "16": "0", "17": "0"})
        self.init_completed = True

    def __repr__(self):
        return self.__class__.__name__

    def getnextID(self):
        ret = self.next_id
        self.next_id += 1
        return ret

    def get_by_id(self, id):
        self.debug("Get by id: %s" % id)
        if id == "0":
            id = "1000"
        try:
            return self.store[id]
        except KeyError:
            self.info("Nothing for %s", id)
            self.debug(list(self.store.keys()))
            return None

    def add_store_item(self, id, item):
        # Working assumption: duplicate ids are the same item
        if id not in self.store:
            self.store[id] = item
        return self.store[id]

    def remove_item(self, item):
        del self.store[item.id]

    def add_new_entry(self, item):
        self.debug("theirs: %s %s", item.__dict__, item.get_id())
        _, ext = os.path.splitext(item.location)
        id = self.getnextID()
        id = str(id)

        try:
            mimetype = KNOWN_AUDIO_TYPES[ext]
        except KeyError:
            mimetype = "audio/mpeg"

        entry = self.add_store_item(
            id,
            ShortListItem(
                id,
                self.root,
                item.location,
                mimetype,
                self.urlbase,
                classChooser(mimetype),
                update=True,
                store=self,
            ),
        )

        entry.item = item.get_item()
        entry.item.title = "%s - %s" % (item.album.artist.name, item.title)

        self.debug(
            "mine %s %s %s", entry, entry.item.__dict__, entry.item.res[0].__dict__
        )
        entry.item_key = str(item.get_id()) + ext
        self.add_store_item(entry.item_key, entry)

        self.root.add_child(entry, update=True)

    def make_playlist(self):
        self.debug("Source backend %s", self.source_backend)
        keys = list(self.source_backend.db.query(Track, sort=Track.title.ascending))
        for x in range(self.trackcount):
            while True:
                if len(keys) == 0:
                    break
                item = random.choice(keys)
                try:
                    self.add_new_entry(item)
                except OSError as e:
                    self.warning("Can't get to %s, got exception %s", item, e)
                    # Can't get to the item in some way, so skip
                    continue
                keys.remove(item)
                break
            if len(keys) == 0:
                break

    def updatePlaylist(self):
        if len(self.root.children) == 0:
            oldest = None
        else:
            oldest = sorted(self.root.children, key=lambda item: int(item.id))[0]
        existing = [int(x.item.id) for x in self.root.children]
        possible = list(self.source_backend.db.query(Track, sort=Track.title.ascending))
        if len(possible) == 0:
            return
        while True:
            item = random.choice(possible)
            if item.get_id() in existing:
                self.debug("duplicate %s", item.get_id())
                continue
            # Don't remove the music in case there's a cached client around
            if oldest is not None:
                self.root.remove_child(oldest)
                self.debug("removed %s %s", oldest.id, oldest.item_key)
            self.debug("adding new %s", item)
            try:
                self.add_new_entry(item)
            except OSError as e:
                # Can't get to the item in some way, so skip
                self.warning("Can't get to %s, got exception %s", item, e)
                continue
            self.update_id += 1
            self.server.content_directory_server.set_variable(
                0, "SystemUpdateID", self.update_id
            )
            self.server.content_directory_server.set_variable(
                0, "ContainerUpdateIDs", (self.root.get_id(), self.root.update_id)
            )
            break

    def upnp_init(self):
        self.source_backend.upnp_init()
        self.debug("upnp_init %s", self.server)
        self.make_playlist()
        loopingCall = task.LoopingCall(self.updatePlaylist)
        loopingCall.start(self.updateFrequency)
        self.current_connection_id = None
        if self.server:
            self.server.connection_manager_server.set_variable(
                0,
                "SourceProtocolInfo",
                [
                    "internal:%s:audio/mpeg:*" % self.server.coherence.hostname,
                    "http-get:*:audio/mpeg:*",
                    "internal:%s:application/ogg:*"
                    % self.server.coherence.hostname,  # noqa
                    "http-get:*:application/ogg:*",
                ],
                default=True,
            )
            self.server.content_directory_server.set_variable(
                0, "SystemUpdateID", self.update_id
            )
            self.server.content_directory_server.set_variable(
                0, "SortCapabilities", "*"
            )


Plugins().set("ShortListStore", ShortListStore)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--music-path", required=True, help="Path to your music files"
)
parser.add_argument("-n", "--name", default="Shortlist", help="Name of UPnP store")
parser.add_argument(
    "-d", "--db", default="music.db", help="Path to music database (default: music.db)"
)
parser.add_argument(
    "-i",
    "--item-count",
    default=50,
    type=int,
    help="Number of tracks in the playlist (default: 50)",
)
parser.add_argument(
    "-u",
    "--update-frequency",
    default=300,
    type=int,
    help="Change out a track every N seconds (default: 300)",
)
args = parser.parse_args()

coherence = Coherence(
    {
        "logging": {
            "level": "warning",
            "subsystem": [
                {"active": "yes", "name": "shortlist_store", "level": "debug"}
            ],
        },
        "controlpoint": "yes",
        "plugin": [
            {
                "backend": "ShortListStore",
                "name": args.name,
                "medialocation": args.music_path,
                "mediadb": args.db,
                "trackcount": args.item_count,
                "updateFrequency": args.update_frequency,
            },
        ],
    }
)

reactor.run()
