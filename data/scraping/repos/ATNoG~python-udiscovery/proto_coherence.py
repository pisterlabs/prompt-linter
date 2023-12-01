from twisted.internet import reactor
from coherence.base import Coherence


def found_device_cb(dev):
    pass



def start():
    config = {'logmode':'warning'}
    c = Coherence(config)
    c.connect(found_device_cb, 'Coherence.UPnP.Device.detection_completed')


raise ImportError # this is not implemented yet

class CoherenceProtocol(BaseProtocol):
    @staticmethod
    def find(uuid_id):
        raise NotImplementedError

    @staticmethod
    def entity2uuid(entity_id):
        raise NotImplementedError

    @staticmethod
    def publish(uuid_id):
        raise NotImplemented

    @staticmethod
    def discover():
        raise NotImplementedError

