"""Pydantic schemas and data transfer objects for microarchitectural inference."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class SingleHardwareConfig(BaseModel):
    workload: str = Field(default="matmul", description="Workload benchmark: matmul, stream, quicksort, hash")
    fetch_width: int = Field(default=4, ge=2, le=8, description="Fetch width (2, 4, 8)")
    decode_width: Optional[int] = Field(default=None, description="Decode width (defaults to fetch_width)")
    issue_width: int = Field(default=4, ge=2, le=8, description="Issue width (2, 4, 8, <= fetch_width)")
    commit_width: Optional[int] = Field(default=None, description="Commit width (defaults to issue_width)")
    rob_entries: int = Field(default=128, ge=32, le=512, description="Reorder Buffer entries")
    iq_entries: Optional[int] = Field(default=None, ge=8, le=128, description="Instruction Queue entries")
    lq_entries: Optional[int] = Field(default=None, ge=8, le=128, description="Load Queue entries")
    sq_entries: Optional[int] = Field(default=None, ge=8, le=128, description="Store Queue entries")
    phys_int_regs: Optional[int] = Field(default=None, ge=32, le=512, description="Physical Integer Registers")
    phys_fp_regs: Optional[int] = Field(default=None, ge=32, le=512, description="Physical Floating-Point Registers")
    branch_predictor: str = Field(default="TournamentBP", description="Branch predictor model")
    l1d_size_kb: int = Field(default=64, ge=8, le=512, description="L1 Data Cache size in KB")
    l1d_assoc: int = Field(default=2, ge=1, le=32, description="L1 Data Cache associativity")
    l1i_size_kb: int = Field(default=32, ge=8, le=512, description="L1 Instruction Cache size in KB")
    l1i_assoc: int = Field(default=2, ge=1, le=32, description="L1 Instruction Cache associativity")
    l2_size_kb: int = Field(default=1024, ge=128, le=8192, description="L2 Unified Cache size in KB")
    l2_assoc: int = Field(default=4, ge=1, le=64, description="L2 Cache associativity")
    clk_freq_ghz: float = Field(default=2.4, ge=1.0, le=5.0, description="Clock frequency in GHz")

class SinglePredictionResponse(BaseModel):
    ipc: float
    uncertainty_sigma: float
    ci_lower: float
    ci_upper: float
    ucb: float
    area_cost_proxy: float
    dynamic_power_proxy: float
    energy_per_instruction_proxy: float
    latency_ms: float
    ratios: Dict[str, float]
    filled_config: Dict[str, Any]

class BatchPredictionResponse(BaseModel):
    total_samples: int
    predictions: List[Dict[str, Any]]
    pareto_frontier: List[Dict[str, Any]]
    summary: Dict[str, Any]
