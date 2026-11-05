"""Decide whether a Queue API base URL may carry service credentials.

HMAC proves who sent a body. It does not hide the body or the bearer token.
Public HTTP is rejected. Loopback, private addresses, and single-label
Docker DNS names may stay on HTTP because they do not cross the public internet.
TLS certificate verification is not disabled here.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

_LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}
_PRIVATE_V4 = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
)
_PRIVATE_V6 = (
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("fc00::/7"),
)


def _is_non_public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    networks = _PRIVATE_V4 if isinstance(address, ipaddress.IPv4Address) else _PRIVATE_V6
    return any(address in network for network in networks)


def queue_api_transport_error(url: str, *, allow_public_http: bool = False) -> str | None:
    """Return a stable error string, or None when the URL is allowed.

    Public HTTP stays rejected unless the process opts in with
    QUEUE_API_ALLOW_PUBLIC_HTTP. TLS verification is unchanged.
    """
    raw = (url or "").strip()
    if not raw:
        return "QUEUE_API_BASE_URL is required"
    parsed = urlparse(raw)
    if parsed.username or parsed.password:
        return "QUEUE_API_BASE_URL must not embed credentials"
    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").strip().lower()
    if not host or parsed.hostname is None:
        return "QUEUE_API_BASE_URL must include a host"
    if scheme == "https":
        return None
    if scheme != "http":
        return "QUEUE_API_BASE_URL must use https, or http only on a private host"
    if host in _LOOPBACK_NAMES:
        return None
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and _is_non_public(address):
        return None
    if address is not None:
        if allow_public_http:
            return None
        return "QUEUE_API_BASE_URL must use https for a public address"
    # queue-api, redis: names resolved only on the compose network.
    if "." not in host:
        return None
    if allow_public_http:
        return None
    return "QUEUE_API_BASE_URL must use https for a public hostname"
