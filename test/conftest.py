"""Provide lightweight stubs for heavy/optional runtime dependencies.

confluent_kafka and scapy are required at runtime but not for unit tests
that exercise pure logic (schema checks, transforms, registry validation).
Stubbing them at collection time lets the entire test suite run in a
minimal environment (no Kafka broker, no packet capture drivers).
"""
import sys
import types
from unittest.mock import MagicMock


def _ensure_mock_module(name: str, attrs: dict | None = None) -> None:
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    mod.__spec__ = None
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod


_ensure_mock_module("confluent_kafka", {
    "Consumer": MagicMock,
    "Producer": MagicMock,
    "KafkaError": type("KafkaError", (Exception,), {}),
})

for _sub in (
    "scapy", "scapy.all", "scapy.arch", "scapy.arch.windows",
):
    _ensure_mock_module(_sub, {
        "sniff": MagicMock(),
        "wrpcap": MagicMock(),
        "rdpcap": MagicMock(),
        "conf": MagicMock(),
        "get_windows_if_list": MagicMock(),
    })

_ensure_mock_module("dotenv", {"load_dotenv": lambda *a, **kw: None})
