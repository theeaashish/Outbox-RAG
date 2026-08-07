from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TypeVar
from uuid import UUID


class InvalidCursorError(ValueError):
    """Raised when a cursor is malformed, invalid, or used outside its scope."""


class CursorResource(StrEnum):
    CONVERSATIONS = "conversations"
    MESSAGES = "messages"


class CursorSort(StrEnum):
    CREATED_AT_DESC_ID_DESC = "created_at_desc_id_desc"


@dataclass(frozen=True, slots=True)
class CursorPosition:
    created_at: datetime
    entity_id: UUID
    snapshot_timestamp: datetime


ModelT = TypeVar("ModelT")


@dataclass(frozen=True, slots=True)
class CursorPage[ModelT]:
    items: list[ModelT]
    has_next_page: bool
    has_previous_page: bool
    snapshot_timestamp: datetime


class CursorCodec:
    """Creates signed, scope-bound cursors for stable keyset pagination."""

    _VERSION = 1
    _DEFAULT_SORT = CursorSort.CREATED_AT_DESC_ID_DESC
    _KEY_ID = "current"

    def __init__(self, *, signing_key: str, previous_signing_key: str | None = None):
        self._signing_key = signing_key.encode()
        self._previous_signing_key = (
            previous_signing_key.encode() if previous_signing_key else None
        )

    @staticmethod
    def _encode_part(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode().rstrip("=")

    @staticmethod
    def _decode_part(value: str) -> bytes:
        try:
            return base64.urlsafe_b64decode(value + ("=" * (-len(value) % 4)))
        except ValueError as exc:
            raise InvalidCursorError("Invalid cursor") from exc

    def encode(
        self,
        *,
        resource: CursorResource,
        scope_id: UUID,
        created_at: datetime,
        entity_id: UUID,
        snapshot_timestamp: datetime,
    ) -> str:
        payload = {
            "v": self._VERSION,
            "kid": self._KEY_ID,
            "r": resource.value,
            "s": str(scope_id),
            "sort": self._DEFAULT_SORT.value,
            "c": created_at.astimezone(UTC).isoformat(),
            "i": str(entity_id),
            "w": snapshot_timestamp.astimezone(UTC).isoformat(),
        }
        encoded_payload = self._encode_part(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        )
        signature = hmac.new(
            self._signing_key,
            encoded_payload.encode(),
            hashlib.sha256,
        ).digest()
        return f"{encoded_payload}.{self._encode_part(signature)}"

    def decode(
        self,
        *,
        cursor: str,
        resource: CursorResource,
        scope_id: UUID,
    ) -> CursorPosition:
        try:
            encoded_payload, encoded_signature = cursor.split(".", maxsplit=1)
            signature = self._decode_part(encoded_signature)
            valid_signature = hmac.compare_digest(
                hmac.new(
                    self._signing_key,
                    encoded_payload.encode(),
                    hashlib.sha256,
                ).digest(),
                signature,
            )
            if not valid_signature and self._previous_signing_key is not None:
                valid_signature = hmac.compare_digest(
                    hmac.new(
                        self._previous_signing_key,
                        encoded_payload.encode(),
                        hashlib.sha256,
                    ).digest(),
                    signature,
                )
            if not valid_signature:
                raise InvalidCursorError("Invalid cursor")

            payload = json.loads(self._decode_part(encoded_payload))
            if (
                payload["v"] != self._VERSION
                or payload["r"] != resource.value
                or payload.get("sort", payload.get("o", self._DEFAULT_SORT.value))
                != self._DEFAULT_SORT.value
                or UUID(payload["s"]) != scope_id
            ):
                raise InvalidCursorError("Invalid cursor")
            return CursorPosition(
                created_at=datetime.fromisoformat(payload["c"]).astimezone(UTC),
                entity_id=UUID(payload["i"]),
                snapshot_timestamp=datetime.fromisoformat(payload["w"]).astimezone(UTC),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, InvalidCursorError):
                raise
            raise InvalidCursorError("Invalid cursor") from exc
