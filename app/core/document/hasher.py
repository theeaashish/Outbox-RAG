from __future__ import annotations

import hashlib


class FileHasher:
    """Utility class for file hashing"""

    def hash(self, content: bytes) -> str:
        """Generate the SHA-256 hash of file content"""
        return hashlib.sha256(content).hexdigest()
