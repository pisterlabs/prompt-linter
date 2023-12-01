from gem5.utils.requires import requires
from gem5.components.boards.x86_board import X86Board
from gem5.components.memory.multi_channel import DualChannelDDR4_2400
from gem5.components.cachehierarchies.ruby.mesi_two_level_cache_hierarchy import MESITwoLevelCacheHierarchy
from gem5.components.cachehierarchies.classic.no_cache import NoCache
from gem5.components.processors.simple_switchable_processor import SimpleSwitchableProcessor
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.coherence_protocol import CoherenceProtocol
from gem5.isas import ISA
from gem5.components.processors.cpu_types import CPUTypes
from gem5.resources.resource import Resource
from gem5.simulate.simulator import Simulator
from gem5.simulate.exit_event import ExitEvent

requires(
	isa_required=ISA.X86,
	coherence_protocol_required=CoherenceProtocol.MESI_TWO_LEVEL,
	kvm_required=False,
	)

cache_hierarchy = MESITwoLevelCacheHierarchy(
	l1d_size="64KiB",
	l1d_assoc=8,
	l1i_size="64KiB",
	l1i_assoc=8,
	l2_size="128MiB",
	l2_assoc=16,
	num_l2_banks=1,
	)

memory = DualChannelDDR4_2400(size="2GiB")


processor = SimpleProcessor(cpu_type=CPUTypes.TIMING, isa=ISA.X86,
                            num_cores=2)

# processor = SimpleSwitchableProcessor(
# 	starting_core_type=CPUTypes.TIMING,
# 	switch_core_type=CPUTypes.TIMING,
#     isa=ISA.X86,
# 	num_cores=2,
# 	)

board = X86Board(
	clk_freq="3GHz",
	processor=processor,
	memory=memory,
	cache_hierarchy=cache_hierarchy,
	)

command = "cd /home/gem5/parsec-benchmark;" \
	    + "source env.sh;" \
	    + "parsecmgmt -a run -p blackscholes -c gcc-hooks -i simsmall -n 2;" \
	    + "sleep 5;" \
	    + "m5 exit;" \

board.set_kernel_disk_workload(
	kernel=Resource("x86-linux-kernel-5.4.49"),
	disk_image=Resource("x86-ubuntu-18.04-img"),
	readfile_contents=command,
	)

simulator = Simulator(
	board=board,
	# on_exit_event={
	# ExitEvent.EXIT : (func() for func in [processor.switch]),
	# },
	)

simulator.run(max_ticks=10**7)

print(
    "Exiting @ tick {} because {}.".format(
        simulator.get_current_tick(),
        simulator.get_last_exit_event_cause(),
    )
)

checkpoint_path = "custom_checkpoint/"
print("Taking a checkpoint at", checkpoint_path)
simulator.save_checkpoint(checkpoint_path)
print("Done taking a checkpoint")
