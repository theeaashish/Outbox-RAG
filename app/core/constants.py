from typing import Any, Final

EMBEDDING_DIMENSION: Final[int] = 768
"""Standard embedding dimension for the project.

All embedding providers must generate vectors with this dimensionality.
Changing this value requires a database migration and re-embedding all
stored document chunks.
"""
JSONDict = dict[str, Any]
