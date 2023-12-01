from m5.objects import DDR4_2400_16x4
from m5.objects import ArmDefaultRelease
from m5.objects import VExpress_GEM5_Foundation

from gem5.isas import ISA
from gem5.utils.override import overrides
from gem5.components.boards.arm_board import ArmBoard
from gem5.components.memory.memory import ChanneledMemory
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.boards.simple_board import SimpleBoard


from ..cmn import CoherentMeshNetwork
from ..processors import (
    SwitchableNovoVectorProcessor,
    NovoVectorProcessor,
    ARM_SVE_Parameters,
)


class NovoverseSystemFS(ArmBoard):
    def __init__(self, clk_freq, num_cores, num_channels, vlen):
        release = ArmDefaultRelease()
        platform = VExpress_GEM5_Foundation()
        sve_parameters = ARM_SVE_Parameters(vlen=vlen, is_fullsystem=True)
        processor = SwitchableNovoVectorProcessor(
            CPUTypes.ATOMIC, num_cores, ISA.ARM, sve_parameters
        )
        memory = ChanneledMemory(
            dram_interface_class=DDR4_2400_16x4,
            num_channels=num_channels,
            interleaving_size=2**8,
            size="16GiB",
        )
        cache_hierarchy = CoherentMeshNetwork()

        super().__init__(
            clk_freq, processor, memory, cache_hierarchy, platform, release
        )

        sve_parameters.apply_system_change(self)

    @overrides(ArmBoard)
    def _pre_instantiate(self):
        super()._pre_instantiate()

    @overrides(ArmBoard)
    def get_default_kernel_args(self):
        return [
            "console=ttyAMA0",
            "lpj=19988480",
            "norandmaps",
            "root=/dev/vda1",
            "rw",
            f"mem={self.get_memory().get_size()}",
            "init=/root/gem5-init.sh",
        ]


class NovoverseSystemSE(SimpleBoard):
    def __init__(self, clk_freq, num_cores, num_channels, vlen):
        sve_parameters = ARM_SVE_Parameters(vlen=vlen, is_fullsystem=False)
        processor = NovoVectorProcessor(num_cores, sve_parameters)
        memory = ChanneledMemory(
            dram_interface_class=DDR4_2400_16x4,
            num_channels=num_channels,
            interleaving_size=2**8,
            size="16GiB",
        )
        cache_hierarchy = CoherentMeshNetwork()

        super().__init__(
            clk_freq,
            processor,
            memory,
            cache_hierarchy,
        )

        sve_parameters.apply_system_change(self)
