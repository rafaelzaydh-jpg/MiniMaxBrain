"""Strict unit parsing used by the physical budget boundary."""
from __future__ import annotations

import re

from .errors import ConfigurationError


_BYTE_UNITS = {
    "b": 1,
    "kb": 1_000,
    "mb": 1_000_000,
    "gb": 1_000_000_000,
    "tb": 1_000_000_000_000,
    "kib": 1 << 10,
    "mib": 1 << 20,
    "gib": 1 << 30,
    "tib": 1 << 40,
}
_COUNT_UNITS = {"": 1, "k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "t": 1_000_000_000_000}


def parse_bytes(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ConfigurationError(f"{field} must be a byte count, not boolean")
    if isinstance(value, int):
        if value < 0:
            raise ConfigurationError(f"{field} must be non-negative")
        return value
    if not isinstance(value, str):
        raise ConfigurationError(f"{field} must be an integer or a size such as 12GiB")
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b)\s*", value, re.IGNORECASE)
    if not match:
        raise ConfigurationError(f"{field} has invalid byte size: {value!r}")
    number = float(match.group(1))
    result = int(number * _BYTE_UNITS[match.group(2).lower()])
    if result < 0:
        raise ConfigurationError(f"{field} must be non-negative")
    return result


def parse_count(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ConfigurationError(f"{field} must be a count, not boolean")
    if isinstance(value, int):
        if value < 0:
            raise ConfigurationError(f"{field} must be non-negative")
        return value
    if not isinstance(value, str):
        raise ConfigurationError(f"{field} must be an integer or a count such as 2T")
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmbt]?)\s*", value, re.IGNORECASE)
    if not match:
        raise ConfigurationError(f"{field} has invalid count: {value!r}")
    return int(float(match.group(1)) * _COUNT_UNITS[match.group(2).lower()])


def format_bytes(value: int) -> str:
    number = float(max(0, int(value)))
    for suffix, divisor in (("TiB", 1 << 40), ("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if number >= divisor:
            return f"{number / divisor:.2f} {suffix}"
    return f"{int(number)} B"

