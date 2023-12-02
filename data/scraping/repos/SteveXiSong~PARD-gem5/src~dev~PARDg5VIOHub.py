# Copyright (c) 2015 Institute of Computing Technology, CAS
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Authors: Jiuyue Ma

from XBar import CoherentXBar
from m5.params import *
from m5.proxy import *
from ControlPlane import ControlPlane
from TagAddrMapper import TagAddrMapper
from XBar import NoncoherentXBar


class PARDg5VIOHubRemapper(TagAddrMapper):
    type = 'PARDg5VIOHubRemapper'
    cxx_header = 'dev/pardg5v_iohub.hh'

    ioh = Param.PARDg5VIOHub(Parent.any, "IOHub this mapper belong to")

class PARDg5VIOHubCP(ControlPlane):
    type = 'PARDg5VIOHubCP'
    cxx_header = 'dev/pardg5v_iohub_cp.hh'

    # CPN address 1:0
    cp_dev = 1
    cp_fun = 0
    # Type 'H' IOHub, IDENT: PARDg5VIOHCP
    Type = 0x48
    IDENT = "PARDg5VIOHCP"

class PARDg5VIOHub(NoncoherentXBar):
    type = 'PARDg5VIOHub'
    cxx_header = 'dev/pardg5v_iohub.hh'

    # PARDg5VIOHub Control Plane
    cp = Param.PARDg5VIOHubCP(PARDg5VIOHubCP(),
                              "Control plane for PARDg5-V IOHub")

    def attachRemappedMaster(self, remapped_master):
        remapped_master.remapper = PARDg5VIOHubRemapper()
        remapped_master.remapper.slave = remapped_master.master
        remapped_master.remapper.master = self.slave
