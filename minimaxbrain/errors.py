"""Typed failures for the external MiniMaxBrain weight gate."""
from __future__ import annotations


class MMBError(RuntimeError):
    """Base failure carrying a stable machine-readable code."""

    code = "MMB_ERROR"

    def __init__(self, detail: str = ""):
        super().__init__(detail or self.code)
        self.detail = detail or self.code


class ManifestError(MMBError):
    code = "MANIFEST_INVALID"


class ConfigurationError(MMBError):
    code = "CONFIG_INVALID"


class BudgetError(MMBError):
    code = "RAM_BUDGET_EXCEEDED"


class IntegrityError(MMBError):
    code = "WEIGHT_INTEGRITY_FAILURE"


class UnknownBlockError(MMBError):
    code = "UNKNOWN_BLOCK"


class AdmissionError(MMBError):
    code = "ADMISSION_FAILED"


class LeaseError(MMBError):
    code = "LEASE_INVALID"


class BackendUnavailableError(MMBError):
    """No real inference backend is ready to generate model output."""

    code = "BACKEND_UNAVAILABLE"


class InferenceError(MMBError):
    """A real backend failed while executing an inference request."""

    code = "INFERENCE_FAILED"

