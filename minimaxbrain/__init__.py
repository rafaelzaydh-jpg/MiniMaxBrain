"""MiniMaxBrain External Gate: bounded admission for independently pageable weights."""
from .client import ExternalGateClient, RemoteLease
from .config import (
    CONFIG_SCHEMA,
    ExternalGateConfig,
    ModelMemoryConfig,
    load_external_bundle,
    load_external_config,
)
from .external import ExternalGate
from .gguf import GGUFModel, GGUFTensor, gguf_summary, load_gguf
from .gguf_moe import (
    GGUF_MOE_BACKEND_CONTRACT,
    GGUF_MOE_LAYOUT_SCHEMA,
    pack_granitemoe_gguf,
    pack_moe_gguf,
)
from .model_map import MODEL_MAP_SCHEMA, PhysicalModelMap, WeightBlock, load_model_map
from .model_memory import MODEL_MEMORY_SCHEMA, ModelMemory
from .packer import PACK_PLAN_SCHEMA, pack_from_plan
from .protocol import IPC_PROTOCOL

__all__ = [
    "CONFIG_SCHEMA",
    "IPC_PROTOCOL",
    "MODEL_MAP_SCHEMA",
    "MODEL_MEMORY_SCHEMA",
    "PACK_PLAN_SCHEMA",
    "ExternalGateConfig",
    "ModelMemoryConfig",
    "ExternalGate",
    "ExternalGateClient",
    "GGUFModel",
    "GGUF_MOE_BACKEND_CONTRACT",
    "GGUF_MOE_LAYOUT_SCHEMA",
    "GGUFTensor",
    "PhysicalModelMap",
    "ModelMemory",
    "RemoteLease",
    "WeightBlock",
    "load_external_bundle",
    "load_external_config",
    "load_gguf",
    "load_model_map",
    "gguf_summary",
    "pack_from_plan",
    "pack_granitemoe_gguf",
    "pack_moe_gguf",
]
