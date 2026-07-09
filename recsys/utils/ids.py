from __future__ import annotations

import hashlib
import uuid


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def pseudonymous_id(raw_identifier: str, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}:{raw_identifier}".encode("utf-8")).hexdigest()
    return digest[:32]
