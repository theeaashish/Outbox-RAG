from __future__ import annotations

from typing import ClassVar

from app.core.ai.context.models import AssembledContext, ContextChunk
from app.core.ai.retrieval.models import RetrievedChunk
from app.core.config import settings
from app.core.exceptions import ValidationException


class ContextAssembler:
    """
    Converts retrieved document chunks into a provider-agnostic
    context package ready for prompt construction.
    """

    MIN_CONTEXT_CHARACTERS: ClassVar[int] = 100

    _SEPARATOR = "\n\n" + ("=" * 80) + "\n\n"
    _TRUNCATION_MARKER = "\n... [truncated to context budget]"
    _EMPTY_CONTEXT_BLOCK = "No relevant context was supplied for this turn."

    def __init__(self, *, max_characters: int | None = None) -> None:
        self._max_characters = (
            max_characters
            if max_characters is not None
            else settings.chat_context_max_characters
        )
        if self._max_characters < self.MIN_CONTEXT_CHARACTERS:
            raise ValidationException(
                f"Context character budget must be at least {self.MIN_CONTEXT_CHARACTERS} characters"
            )

    def assemble(
        self,
        *,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        max_characters: int | None = None,
    ) -> AssembledContext:
        """
        Assemble retrieved chunks into a prompt-ready context package.

        Applies the deterministic pipeline:
        retrieve -> deduplicate -> budget/filter -> assign final sequential citations -> build context block.
        """
        effective_budget = (
            max_characters if max_characters is not None else self._max_characters
        )
        if effective_budget < self.MIN_CONTEXT_CHARACTERS:
            raise ValidationException(
                f"Context character budget must be at least {self.MIN_CONTEXT_CHARACTERS} characters"
            )

        deduplicated = self._deduplicate_chunks(retrieved_chunks)
        if not deduplicated:
            return AssembledContext(
                query=query,
                block=self._EMPTY_CONTEXT_BLOCK,
                chunks=[],
            )

        context_chunks, block = self._budget_and_render(
            deduplicated=deduplicated,
            budget=effective_budget,
        )

        return AssembledContext(
            query=query,
            block=block,
            chunks=context_chunks,
        )

    def _deduplicate_chunks(
        self,
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Deduplicate chunks by (document_id, chunk_index), retaining the one with highest similarity.
        Preserves similarity-descending order.
        """
        best_by_key: dict[tuple[object, int], RetrievedChunk] = {}

        for item in retrieved_chunks:
            key = (item.chunk.document_id, item.chunk.chunk_index)
            existing = best_by_key.get(key)
            if existing is None or item.similarity > existing.similarity:
                best_by_key[key] = item

        return sorted(best_by_key.values(), key=lambda x: x.similarity, reverse=True)

    def _format_source_header(
        self,
        *,
        citation: int,
        document_name: str,
        page_start: int | None,
        page_end: int | None,
    ) -> str:
        """Format the structured header for a single context source."""
        if page_start is None:
            page_suffix = ""
        elif page_start == page_end:
            page_suffix = f", Page {page_start}"
        else:
            page_suffix = f", Pages {page_start}-{page_end}"

        return f'[Source {citation}] Document: "{document_name}"{page_suffix}\n---\n'

    @staticmethod
    def _sanitize_for_serialization(content: str) -> str:
        """Escape context delimiters to prevent serialization collisions."""
        return content.replace("</retrieved_context>", r"<\/retrieved_context>")

    def _budget_and_render(
        self,
        *,
        deduplicated: list[RetrievedChunk],
        budget: int,
    ) -> tuple[list[ContextChunk], str]:
        """
        Fit deduplicated chunks into the character budget and assign sequential citations.

        The budget applies to the complete rendered block (headers, separators, markers, and content).
        """
        first_item = deduplicated[0]
        first_doc_name = (
            first_item.chunk.document.title
            if first_item.chunk.document and first_item.chunk.document.title
            else "Untitled"
        )
        first_content = self._sanitize_for_serialization(first_item.chunk.content)
        first_provenance = first_item.provenance

        first_header = self._format_source_header(
            citation=1,
            document_name=first_doc_name,
            page_start=first_provenance.page_start,
            page_end=first_provenance.page_end,
        )
        first_rendered = first_header + first_content

        if len(first_rendered) > budget:
            overhead = len(first_header) + len(self._TRUNCATION_MARKER)
            available_chars = max(0, budget - overhead)
            truncated_content = first_content[:available_chars]
            final_block = first_header + truncated_content + self._TRUNCATION_MARKER

            chunk = ContextChunk(
                citation=1,
                document_id=first_item.chunk.document_id,
                document_name=first_doc_name,
                chunk_index=first_item.chunk.chunk_index,
                similarity=first_item.similarity,
                content=truncated_content,
                page_start=first_provenance.page_start,
                page_end=first_provenance.page_end,
            )
            return [chunk], final_block

        included_chunks: list[ContextChunk] = [
            ContextChunk(
                citation=1,
                document_id=first_item.chunk.document_id,
                document_name=first_doc_name,
                chunk_index=first_item.chunk.chunk_index,
                similarity=first_item.similarity,
                content=first_content,
                page_start=first_provenance.page_start,
                page_end=first_provenance.page_end,
            )
        ]
        rendered_sections: list[str] = [first_rendered]
        current_len = len(first_rendered)

        for item in deduplicated[1:]:
            next_citation = len(included_chunks) + 1
            doc_name = (
                item.chunk.document.title
                if item.chunk.document and item.chunk.document.title
                else "Untitled"
            )
            content = self._sanitize_for_serialization(item.chunk.content)
            provenance = item.provenance

            header = self._format_source_header(
                citation=next_citation,
                document_name=doc_name,
                page_start=provenance.page_start,
                page_end=provenance.page_end,
            )
            rendered_chunk = header + content
            additional_len = len(self._SEPARATOR) + len(rendered_chunk)

            if current_len + additional_len <= budget:
                included_chunks.append(
                    ContextChunk(
                        citation=next_citation,
                        document_id=item.chunk.document_id,
                        document_name=doc_name,
                        chunk_index=item.chunk.chunk_index,
                        similarity=item.similarity,
                        content=content,
                        page_start=provenance.page_start,
                        page_end=provenance.page_end,
                    )
                )
                rendered_sections.append(rendered_chunk)
                current_len += additional_len
            else:
                break

        return included_chunks, self._SEPARATOR.join(rendered_sections)
