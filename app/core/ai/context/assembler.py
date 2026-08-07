from __future__ import annotations

from app.core.ai.context.models import AssembledContext, ContextChunk
from app.core.ai.retrieval.models import RetrievedChunk


class ContextAssembler:
    """
    Converts retrieved document chunks into a provider-agnostic
    context package ready for prompt construction.
    """

    _SEPARATOR = "\n\n" + ("-" * 80) + "\n\n"

    def assemble(
        self,
        *,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> AssembledContext:
        """
        Assemble retrieved chunks into a prompt-ready context package.
        """

        context_chunks = self._to_context_chunks(retrieved_chunks)

        block = self._build_context_block(context_chunks)

        return AssembledContext(
            query=query,
            block=block,
            chunks=context_chunks,
        )

    def _to_context_chunks(
        self,
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[ContextChunk]:
        """
        Convert retrieval models into provider-agnostic context chunks.
        """

        return [
            ContextChunk(
                citation=index,
                document_id=item.chunk.document_id,
                document_name=item.chunk.document.title,
                chunk_index=item.chunk.chunk_index,
                similarity=item.similarity,
                content=item.chunk.content,
            )
            for index, item in enumerate(retrieved_chunks, start=1)
        ]

    def _build_context_block(
        self,
        chunks: list[ContextChunk],
    ) -> str:
        """
        Build the formatted context block supplied to the prompt builder.
        """

        sections: list[str] = []

        for chunk in chunks:
            sections.append(
                f"[{chunk.citation}] {chunk.document_name}\n\n{chunk.content}"
            )

        return self._SEPARATOR.join(sections)
