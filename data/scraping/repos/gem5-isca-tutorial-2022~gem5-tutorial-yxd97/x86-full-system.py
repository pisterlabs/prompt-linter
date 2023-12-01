from setuptools import Command
from gem5.utils.requires import requires
from gem5.components.boards.x86_board import X86Board
from gem5.components.memory.single_channel import SingleChannelDDR3_1600
from gem5.components.cachehierarchies.ruby.mesi_two_level_cache_hierarchy import MESITwoLevelCacheHierarchy
from gem5.components.processors.simple_switchable_processor import SimpleSwitchableProcessor
from gem5.coherence_protocol import CoherenceProtocol
from gem5.isas import ISA
from gem5.components.processors.cpu_types import CPUTypes
from gem5.resources.resource import Resource
from gem5.simulate.simulator import Simulator
from gem5.simulate.exit_event import ExitEvent

# sanity check before running simulation
requires(
    isa_required=ISA.X86,
    coherence_protocol_required=CoherenceProtocol.MESI_TWO_LEVEL,
)

cache_hier = MESITwoLevelCacheHierarchy(
    l1d_size = "32KiB",
    l1d_assoc = 8,
    l1i_size = "32KiB",
    l1i_assoc = 8,
    l2_size = "256kB",
    l2_assoc = 16,
    num_l2_banks = 1,
)

mem = SingleChannelDDR3_1600("2GiB")

proc = SimpleSwitchableProcessor(
    starting_core_type=CPUTypes.TIMING,
    switch_core_type=CPUTypes.O3,
    num_cores=2,
)

board = X86Board(
    clk_freq = "3GHz",
    processor = proc,
    memory = mem,
    cache_hierarchy = cache_hier,
)

cmd = "m5 exit;" \
    + "echo 'This is running on Timing CPU cores.';" \
    + "sleep 1;" \
    + "m5 exit;"    

board.set_kernel_disk_workload(
    kernel = Resource("x86-linux-kernel-5.4.49"),
    disk_image = Resource("x86-ubuntu-18.04-img"),
    readfile_contents = cmd
)

sim = Simulator(
    board = board,
    on_exit_event = {
        ExitEvent.EXIT : (func() for func in [proc.switch])
    }
)

sim.run()