"""Protocol support

The notion of protocol is slightly skewed, for example
there may be multiple _Protocols_ for bluetooth. Protocols
work like plugins that ensure that you can use this
seamlessly with existing implementations.

The end goal here is that the protocol implementations
return the same data types as the original implementation.
"""

from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division
from abc import abstractmethod

PROTOCOLS = {}

class BaseProtocol(object):
    """The base class for all protocols"""
    @staticmethod
    @abstractmethod
    def find(uuid_id):
        """
        Find entity for the given uuid

        uuid_id: a uuid.UUID object
        """
        pass
    @staticmethod
    @abstractmethod
    def publish(uuid_id, channel=None):
        """
        Advertise uuid

        uuid_id: a uuid.UUID object
        """
        pass
    @staticmethod
    @abstractmethod
    def entity2uuid(entity_id):
        """
        From the given protocol specific id yield a list
        of corresponding UUID objects
        """
        pass

    @staticmethod
    @abstractmethod
    def discover():
        """
        Discover all nearby entities

        returns a dict mapping entitiy identifiers into uuid objects
        """
        pass

# This allows our plugins to be plugable
# i.e. if a dependency is not available then it
# is not loaded
try:
    from .proto_pybluez import PyBluezProtocol
    PROTOCOLS['pybluez'] = PyBluezProtocol
except ImportError:
    pass

try:
    from .proto_pybonjour import PyBonjourProtocol
    PROTOCOLS['pybonjour'] = PyBonjourProtocol
except ImportError:
    pass

try:
    # Coherence provides DLNA/UPnP
    from .proto_conherence import CoherenceProtocol
except ImportError:
    pass




