"""保护文档推荐事实并限制最终返回数量。"""

from __future__ import annotations

from collections.abc import Iterable

from app.models.article import DocumentCandidate, RankedDocument


class DocumentResultAggregator:
    """保护 SQLite 文档候选事实并限制最终推荐数量。"""

    def aggregate(
        self,
        *,
        candidates: list[DocumentCandidate],
        ranked_documents: list[RankedDocument],
        seen_document_ids: Iterable[str],
        size: int,
    ) -> list[RankedDocument]:
        """拒绝越权、重复和已展示文档，并从召回候选恢复真实字段。"""

        if not 0 <= size <= 10:
            raise ValueError("推荐数量必须位于 0 到 10 之间")
        safe_candidates = [
            DocumentCandidate.model_validate(item).model_copy(deep=True)
            for item in candidates
        ]
        candidate_by_id = {item.document_id: item for item in safe_candidates}
        if len(candidate_by_id) != len(safe_candidates):
            raise ValueError("候选 document_id 重复")
        seen = {
            value.strip()
            for value in seen_document_ids
            if isinstance(value, str) and value.strip()
        }
        result: list[RankedDocument] = []
        selected: set[str] = set()
        for raw in ranked_documents:
            ranked = RankedDocument.model_validate(raw)
            candidate = candidate_by_id.get(ranked.document_id)
            if (
                candidate is None
                or ranked.document_id in seen
                or ranked.document_id in selected
            ):
                continue
            result.append(
                RankedDocument(
                    **candidate.model_dump(),
                    llm_score=ranked.llm_score,
                    relevance_score=ranked.relevance_score,
                    profile_score=ranked.profile_score,
                    length_level=ranked.length_level,
                    final_score=ranked.final_score,
                    rerank_reason=ranked.rerank_reason,
                )
            )
            selected.add(ranked.document_id)
            if len(result) >= size:
                break
        return result


__all__ = ["DocumentResultAggregator"]
