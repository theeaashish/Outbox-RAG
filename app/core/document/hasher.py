from __future__ import annotations

import hashlib


class FileHasher:
    """Utility class for file hashing"""

    @staticmethod
    def sha256(content: bytes) -> str:
        """generate the SHA-256 hash of file content"""
        return hashlib.sha256(content).hexdigest()
