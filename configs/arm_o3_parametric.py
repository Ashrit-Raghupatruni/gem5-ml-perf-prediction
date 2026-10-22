"""
gem5 Standard Library: Parametric ARM Out-of-Order (O3) Processor Simulation Engine
Target Architecture: ARM AArch64
Strictly adheres to: documents/architecture.md, documents/schema.md
"""

import argparse
import sys
from gem5.isas import ISA
from gem5.utils.requires import requires
from gem5.utils.override import overrides
from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.boards.abstract_board import AbstractBoard
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.memory.single_channel import SingleChannelDDR4_2400
from gem5.components.cachehierarchies.abstract_two_level_cache_hierarchy import AbstractTwoLevelCacheHierarchy
from gem5.components.cachehierarchies.classic.abstract_classic_cache_hierarchy import AbstractClassicCacheHierarchy
from gem5.components.cachehierarchies.classic.caches.l1dcache import L1DCache
from gem5.components.cachehierarchies.classic.caches.l1icache import L1ICache
from gem5.components.cachehierarchies.classic.caches.l2cache import L2Cache
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator
from m5.objects import (
    BadAddr, BaseCPU, BaseXBar, Cache, L2XBar, SystemXBar, Port,
    TournamentBP, BiModeBP, LocalBP
)

# Enforce ARM ISA
requires(isa_required=ISA.ARM)

parser = argparse.ArgumentParser(description="Parametric ARM O3 Simulation Engine")
parser.add_argument("--binary", type=str, required=True, help="Path to static ARM64 binary")
parser.add_argument("--width", type=int, default=4, choices=[2, 4, 8], help="Pipeline width")
parser.add_argument("--rob", type=int, default=128, help="Number of ROB entries (e.g. 64, 96, 128, 192, 256)")
parser.add_argument("--iq", type=int, default=64, help="Instruction Queue entries")
parser.add_argument("--lq", type=int, default=32, help="Load Queue entries")
parser.add_argument("--sq", type=int, default=32, help="Store Queue entries")
parser.add_argument("--phys-regs", type=int, default=192, help="Physical registers (Int/FP)")
parser.add_argument("--branch-pred", type=str, default="TournamentBP", choices=["TournamentBP", "BiModeBP", "LocalBP"])

parser.add_argument("--l1d-size", type=str, default="64KiB", help="L1D cache size")
parser.add_argument("--l1d-assoc", type=int, default=4, help="L1D cache associativity")
parser.add_argument("--l1i-size", type=str, default="64KiB", help="L1I cache size")
parser.add_argument("--l1i-assoc", type=int, default=4, help="L1I cache associativity")
parser.add_argument("--l2-size", type=str, default="1024KiB", help="L2 cache size")
parser.add_argument("--l2-assoc", type=int, default=8, help="L2 cache associativity")
parser.add_argument("--clk-freq", type=str, default="2.4GHz", help="Clock frequency")

args = parser.parse_args()


class ParametricTwoLevelCacheHierarchy(AbstractClassicCacheHierarchy, AbstractTwoLevelCacheHierarchy):
    def __init__(self, l1i_size, l1i_assoc, l1d_size, l1d_assoc, l2_size, l2_assoc):
        AbstractClassicCacheHierarchy.__init__(self=self)
        AbstractTwoLevelCacheHierarchy.__init__(
            self,
            l1i_size=l1i_size,
            l1i_assoc=l1i_assoc,
            l1d_size=l1d_size,
            l1d_assoc=l1d_assoc,
            l2_size=l2_size,
            l2_assoc=l2_assoc,
        )
        self.membus = SystemXBar(width=64)
        self.membus.badaddr_responder = BadAddr()
        self.membus.default = self.membus.badaddr_responder.pio

    @overrides(AbstractClassicCacheHierarchy)
    def get_mem_side_port(self) -> Port:
        return self.membus.mem_side_ports

    @overrides(AbstractClassicCacheHierarchy)
    def get_cpu_side_port(self) -> Port:
        return self.membus.cpu_side_ports

    @overrides(AbstractClassicCacheHierarchy)
    def incorporate_cache(self, board: AbstractBoard) -> None:
        board.connect_system_port(self.membus.cpu_side_ports)
        for _, port in board.get_mem_ports():
            self.membus.mem_side_ports = port

        self.l2buses = [L2XBar() for _ in range(board.get_processor().get_num_cores())]

        for i, cpu in enumerate(board.get_processor().get_cores()):
            l2_node = self.add_root_child(
                f"l2-cache-{i}",
                L2Cache(size=self._l2_size, assoc=self._l2_assoc, tag_latency=9, data_latency=9)
            )
            l1i_node = l2_node.add_child(
                f"l1i-cache-{i}",
                L1ICache(size=self._l1i_size, assoc=self._l1i_assoc, tag_latency=1, data_latency=1)
            )
            l1d_node = l2_node.add_child(
                f"l1d-cache-{i}",
                L1DCache(size=self._l1d_size, assoc=self._l1d_assoc, tag_latency=3, data_latency=3)
            )

            self.l2buses[i].mem_side_ports = l2_node.cache.cpu_side
            self.membus.cpu_side_ports = l2_node.cache.mem_side

            l1i_node.cache.mem_side = self.l2buses[i].cpu_side_ports
            l1d_node.cache.mem_side = self.l2buses[i].cpu_side_ports

            cpu.connect_icache(l1i_node.cache.cpu_side)
            cpu.connect_dcache(l1d_node.cache.cpu_side)

            cpu.connect_walker_ports(
                self.l2buses[i].cpu_side_ports,
                self.l2buses[i].cpu_side_ports,
            )
            cpu.connect_interrupt()


# 1. Setup cache hierarchy
cache_hierarchy = ParametricTwoLevelCacheHierarchy(
    l1i_size=args.l1i_size,
    l1i_assoc=args.l1i_assoc,
    l1d_size=args.l1d_size,
    l1d_assoc=args.l1d_assoc,
    l2_size=args.l2_size,
    l2_assoc=args.l2_assoc,
)

# 2. Setup memory system
memory = SingleChannelDDR4_2400(size="2GiB")

# 3. Setup ARM Out-of-Order Processor
processor = SimpleProcessor(
    cpu_type=CPUTypes.O3,
    isa=ISA.ARM,
    num_cores=1,
)

# Configure internal O3 pipeline and buffer parameters
for core in processor.get_cores():
    sim_core = core.get_simobject()
    
    # Configure Pipeline Widths
    sim_core.fetchWidth = args.width
    sim_core.decodeWidth = args.width
    sim_core.renameWidth = args.width
    sim_core.issueWidth = args.width
    sim_core.dispatchWidth = args.width
    sim_core.wbWidth = args.width
    sim_core.commitWidth = args.width
    
    # Configure Execution Buffers
    sim_core.numROBEntries = args.rob
    sim_core.numIQEntries = args.iq
    sim_core.LQEntries = args.lq
    sim_core.SQEntries = args.sq
    sim_core.numPhysIntRegs = args.phys_regs
    sim_core.numPhysFloatRegs = args.phys_regs
    
    # Configure Branch Predictor
    if args.branch_pred == "TournamentBP":
        sim_core.branchPred = TournamentBP()
    elif args.branch_pred == "BiModeBP":
        sim_core.branchPred = BiModeBP()
    elif args.branch_pred == "LocalBP":
        sim_core.branchPred = LocalBP()

# 4. Create simple board
board = SimpleBoard(
    clk_freq=args.clk_freq,
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)

# 5. Set binary workload
board.set_se_binary_workload(BinaryResource(local_path=args.binary))

# 6. Run simulation
simulator = Simulator(board=board)
print(f"=== Starting ARM O3 Simulation: Width={args.width}, ROB={args.rob}, Freq={args.clk_freq} ===")
simulator.run()
print("=== Finished ARM O3 Simulation Successfully ===")
