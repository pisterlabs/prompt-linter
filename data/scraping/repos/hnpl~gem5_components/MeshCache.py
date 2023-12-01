from math import log2
from typing import List

from gem5.utils.requires import requires
from gem5.utils.override import overrides
from gem5.coherence_protocol import CoherenceProtocol
from gem5.components.boards.abstract_board import AbstractBoard

from gem5.components.cachehierarchies.ruby.abstract_ruby_cache_hierarchy import AbstractRubyCacheHierarchy
from gem5.components.cachehierarchies.abstract_three_level_cache_hierarchy import AbstractThreeLevelCacheHierarchy
from gem5.components.cachehierarchies.abstract_cache_hierarchy import AbstractCacheHierarchy
from gem5.components.cachehierarchies.chi.nodes.dma_requestor import DMARequestor
from gem5.components.cachehierarchies.chi.nodes.memory_controller import MemoryController
from gem5.components.cachehierarchies.chi.nodes.abstract_node import AbstractNode

from m5.objects import RubySystem, RubyPortProxy, RubySequencer, AddrRange

from .components.CoreTile import CoreTile
from .components.DMATile import DMATile
from .components.L3OnlyTile import L3OnlyTile
from .components.L3Slice import L3Slice
from .components.MemTile import MemTile
from .components.MeshDescriptor import MeshTracker, NodeType
from .components.MeshNetwork import MeshNetwork
from .utils.SizeArithmetic import SizeArithmetic

class MeshCache(AbstractRubyCacheHierarchy, AbstractThreeLevelCacheHierarchy):
    def __init__(
        self,
        l1i_size: str,
        l1i_assoc: int,
        l1d_size: str,
        l1d_assoc: int,
        l2_size: str,
        l2_assoc: int,
        l3_size: str,
        l3_assoc: int,
        num_core_complexes: int,
        is_fullsystem: bool,
        mesh_descriptor: MeshTracker
    ):
        AbstractRubyCacheHierarchy.__init__(self=self)
        AbstractThreeLevelCacheHierarchy.__init__(
            self=self,
            l1i_size=l1i_size,
            l1i_assoc=l1i_assoc,
            l1d_size=l1d_size,
            l1d_assoc=l1d_assoc,
            l2_size=l2_size,
            l2_assoc=l2_assoc,
            l3_size=l3_size,
            l3_assoc=l3_assoc,
        )

        self._num_core_complexes = num_core_complexes
        self._is_fullsystem = is_fullsystem
        self._mesh_descriptor = mesh_descriptor
        self._has_dma = False
        self._has_l3_only_tiles = False

        print(self._mesh_descriptor)

        requires(coherence_protocol_required=CoherenceProtocol.CHI)

    @overrides(AbstractCacheHierarchy)
    def incorporate_cache(self, board: AbstractBoard) -> None:
        self._setup_ruby_system()
        self._get_board_info(board)

        self._create_core_tiles(board)
        self._create_l3_only_tiles(board)
        self._assign_addr_range(board)
        self._create_memory_tiles(board)
        self._create_dma_tiles(board)
        self._set_downstream_destinations()
        self.ruby_system.network.create_mesh()
        self._incorperate_system_ports(board)

        self._finalize_ruby_system()

    def _get_board_info(self, board: AbstractBoard) -> None:
        self._cache_line_size = board.cache_line_size
        self._clk_domain = board.clk_domain

    # should be called at the BEGINNING of incorporate_cache()
    def _setup_ruby_system(self) -> None:
        self.ruby_system = RubySystem()
        self.ruby_system.number_of_virtual_networks = 4
        self.ruby_system.network = MeshNetwork(
            ruby_system = self.ruby_system,
            mesh_descriptor = self._mesh_descriptor
        )
        self.ruby_system.network.number_of_virtual_networks = 4
        self.ruby_system.num_of_sequencers = 0

    # should be called at the END of incorporate_cache()
    def _finalize_ruby_system(self) -> None:
        self.ruby_system.num_of_sequencers = self.ruby_system.network.get_num_sequencers()
        self.ruby_system.network.int_links = self.ruby_system.network._int_links
        self.ruby_system.network.ext_links = self.ruby_system.network._ext_links
        self.ruby_system.network.routers = self.ruby_system.network._routers
        self.ruby_system.network.setup_buffers()

    def _create_core_tiles(self, board: AbstractBoard) -> None:
        core_tile_coordinates = self._mesh_descriptor.get_tiles_coordinates(NodeType.CoreTile)
        cores = board.get_processor().get_cores()
        num_l3_slices = self._mesh_descriptor.get_num_l3_slices()
        l3_slice_size = (SizeArithmetic(self._l3_size) // num_l3_slices).get()
        self.core_tiles = [CoreTile(
            board = board,
            ruby_system = self.ruby_system,
            coordinate = core_tile_coordinate,
            mesh_descriptor = self._mesh_descriptor,
            core = core,
            core_id = core_id//2,
            l1i_size = self._l1i_size,
            l1i_associativity = self._l1i_assoc,
            l1d_size = self._l1d_size,
            l1d_associativity = self._l1d_assoc,
            l2_size = self._l2_size,
            l2_associativity = self._l2_assoc,
            l3_slice_size = l3_slice_size,
            l3_associativity = self._l3_assoc
        ) for core_id, (core, core_tile_coordinate) in enumerate(zip(cores, core_tile_coordinates))]
        for tile in self.core_tiles:
            self.ruby_system.network.incorporate_ruby_subsystem(tile)

    def _create_l3_only_tiles(self, board: AbstractBoard) -> None:
        l3_only_tiles_coordinates = self._mesh_descriptor.get_tiles_coordinates(NodeType.L3OnlyTile)
        num_l3_slices = self._mesh_descriptor.get_num_l3_slices()
        l3_slice_size = (SizeArithmetic(self._l3_size) // num_l3_slices).get()
        if len(l3_only_tiles_coordinates) > 0:
            self._has_l3_only_tiles = True
            self.l3_only_tiles = [L3OnlyTile(
                board = board,
                ruby_system = self.ruby_system,
                coordinate = tile_coordinate,
                mesh_descriptor = self._mesh_descriptor,
                l3_slice_size = l3_slice_size,
                l3_associativity = self._l3_assoc
            ) for tile_coordinate in l3_only_tiles_coordinates]
            for tile in self.l3_only_tiles:
                self.ruby_system.network.incorporate_ruby_subsystem(tile)

    def _find_board_mem_start(self, board: AbstractBoard) -> None:
        mem_start = 1 << 64
        for r in board.mem_ranges:
            mem_start = min(r.start.value, mem_start)
        return mem_start

    def _assign_addr_range(self, board: AbstractBoard) -> None:
        #mem_start = board.get_memory().get_start_addr()
        mem_start = self._find_board_mem_start(board)
        mem_size = board.get_memory().get_size()
        interleaving_size = "64B"
        num_offset_bits = int(log2(SizeArithmetic(interleaving_size).bytes))
        all_l3_slices = self._get_all_l3_slices()
        num_l3_slices = len(all_l3_slices)
        num_slice_indexing_bits = int(log2(num_l3_slices))
        address_ranges = [AddrRange(
            start = mem_start,
            size = mem_size,
            intlvHighBit = num_offset_bits + num_slice_indexing_bits - 1,
            intlvBits = num_slice_indexing_bits,
            intlvMatch = i
        ) for i in range(num_l3_slices)]
        for address_range, l3_slice in zip(address_ranges, all_l3_slices):
            l3_slice.addr_ranges = address_range

    def _create_memory_tiles(self, board: AbstractBoard) -> None:
        mem_tile_coordinates = self._mesh_descriptor.get_tiles_coordinates(NodeType.MemTile)
        self.memory_tiles = [MemTile(
            board = board,
            ruby_system = self.ruby_system,
            coordinate = mem_tile_coordinate,
            mesh_descriptor = self._mesh_descriptor,
            address_range = address_range,
            memory_port = memory_port
        ) for mem_tile_coordinate, (address_range, memory_port) in zip(mem_tile_coordinates, board.get_mem_ports())]
        for tile in self.memory_tiles:
            self.ruby_system.network.incorporate_ruby_subsystem(tile)

    def _create_dma_tiles(self, board: AbstractBoard) -> None:
        self._has_dma = False
        if not board.has_dma_ports():
            return
        self._has_dma = True
        dma_tile_coordinates = self._mesh_descriptor.get_tiles_coordinates(NodeType.DMATile)
        self.dma_tiles = [DMATile(
            board = board,
            ruby_system = self.ruby_system,
            coordinate = dma_tile_coordinate,
            mesh_descriptor = self._mesh_descriptor,
            dma_port = dma_port,
            dma_id = dma_id
        ) for dma_id, (dma_tile_coordinate, dma_port) in enumerate(zip(dma_tile_coordinates, board.get_dma_ports()))]
        for tile in self.dma_tiles:
            self.ruby_system.network.incorporate_ruby_subsystem(tile)

    def _get_all_l3_slices(self) -> List[L3Slice]:
        if self._has_l3_only_tiles:
            all_l3_slices = [tile.l3_slice for tile in self.core_tiles] + [tile.l3_slice for tile in self.l3_only_tiles]
        else:
            all_l3_slices = [tile.l3_slice for tile in self.core_tiles]
        return all_l3_slices

    def _set_downstream_destinations(self) -> None:
        all_l3_slices = self._get_all_l3_slices()
        all_mem_ctrls = [mem_tile.memory_controller for mem_tile in self.memory_tiles]
        for tile in self.core_tiles:
            tile.set_l2_downstream_destinations(all_l3_slices)
        for l3_slice in all_l3_slices:
            l3_slice.downstream_destinations = all_mem_ctrls
        if self._has_dma:
            for tile in self.dma_tiles:
                tile.dma_controller.downstream_destinations = all_l3_slices

    def _incorperate_system_ports(self, board: AbstractBoard) -> None:
        self.ruby_system.sys_port_proxy = RubyPortProxy()
        board.connect_system_port(self.ruby_system.sys_port_proxy.in_ports)
