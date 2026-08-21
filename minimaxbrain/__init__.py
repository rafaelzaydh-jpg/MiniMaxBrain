"""Public MiniMaxBrain API.

Only the verified pager, GGUF converter and fail-closed runtime are exported.
Removed experimental/legacy subsystems are intentionally not part of the
package surface.
"""
from .config import CONFIG_SCHEMA, ExternalGateConfig, load_external_bundle, load_external_config
from .external import ExternalGate
from .gguf import GGUFModel, GGUFTensor, gguf_summary, load_gguf
from .gguf_moe import GGUF_MOE_BACKEND_CONTRACT, GGUF_MOE_LAYOUT_SCHEMA, pack_moe_gguf
from .model_map import MODEL_MAP_SCHEMA, PhysicalModelMap, WeightBlock, load_model_map
from .native import NativePager, NativeSegment, find_native_backend
from .runtime import InferenceMode, MMBRuntime
from .storage import create_model_seal, verify_model_seal

__all__ = [
    "CONFIG_SCHEMA",
    "MODEL_MAP_SCHEMA",
    "ExternalGateConfig",
    "ExternalGate",
    "GGUFModel",
    "GGUFTensor",
    "GGUF_MOE_BACKEND_CONTRACT",
    "GGUF_MOE_LAYOUT_SCHEMA",
    "PhysicalModelMap",
    "WeightBlock",
    "InferenceMode",
    "MMBRuntime",
    "NativePager",
    "NativeSegment",
    "load_external_bundle",
    "load_external_config",
    "load_gguf",
    "load_model_map",
    "gguf_summary",
    "pack_moe_gguf",
    "create_model_seal",
    "verify_model_seal",
    "find_native_backend",
]
