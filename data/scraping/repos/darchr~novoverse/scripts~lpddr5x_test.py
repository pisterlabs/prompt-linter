from util import novoverse


@novoverse
def run_lpddr5x_test():
    import m5

    from m5.objects import Root

    from gem5.components.boards.test_board import TestBoard
    from gem5.components.memory.memory import ChanneledMemory
    from gem5.components.processors.linear_generator import LinearGenerator

    from components.memory import LPDDR5X
    from components.cmn import CoherentMeshNetwork
    from components.strided_generator import StridedGenerator

    memory = ChanneledMemory(
        dram_interface_class=LPDDR5X, num_channels=1, interleaving_size=128
    )

    generator = StridedGenerator(
        num_cores=1,
        duration="100us",
        rate="64GB/s",
        block_size=64,
        superblock_size=512,
        stride_size=4096,
        min_addr=0,
        max_addr=memory.get_size(),
        data_limit=2048,
    )

    board = TestBoard(
        clk_freq="4GHz",
        generator=generator,
        memory=memory,
        cache_hierarchy=None,
    )

    root = Root(full_system=False, system=board)

    board._pre_instantiate()
    m5.instantiate()

    generator.start_traffic()
    print("Beginning simulation!")

    exit_event = m5.simulate()
    print(f"Exiting @ tick {m5.curTick()} because {exit_event.getCause()}.")


if __name__ == "__m5_main__":
    run_lpddr5x_test()
