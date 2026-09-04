"""根据当前快照中的受控证据关系确定性检查计划覆盖。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.models.knowledge_qa import (
    KnowledgeChunkRecord,
    KnowledgePlanCoverage,
    KnowledgePlanEvidenceRelation,
    KnowledgePlanReasonCode,
    KnowledgePlanStep,
    KnowledgePlanStepResult,
    KnowledgeReasoningPlan,
)


_EMPTY_REASON_CODES = frozenset(
    ("no_hits", "scope_filtered", "stale_candidates", "search_failed")
)


class KnowledgePlanCoverageChecker:
    """只依据严格 DTO 和 SQLite 回查快照计算步骤覆盖与最终动作。"""

    def evaluate(
        self,
        plan: KnowledgeReasoningPlan,
        *,
        relations: Sequence[KnowledgePlanEvidenceRelation],
        records: Sequence[KnowledgeChunkRecord],
        empty_reason_by_step: Mapping[str, KnowledgePlanReasonCode],
        replanned: bool,
        allow_replan: bool,
    ) -> KnowledgePlanCoverage:
        """返回每步状态、严格覆盖率以及回答或重规划决策。"""

        normalized_plan = KnowledgeReasoningPlan.model_validate(plan).model_copy(
            deep=True
        )
        normalized_records = tuple(
            KnowledgeChunkRecord.model_validate(record).model_copy(deep=True)
            for record in records
        )
        normalized_relations = tuple(
            KnowledgePlanEvidenceRelation.model_validate(relation).model_copy(
                deep=True
            )
            for relation in relations
        )
        if not isinstance(replanned, bool) or not isinstance(allow_replan, bool):
            raise ValueError("覆盖检查的轮次状态必须是布尔值")
        records_by_id = self._records_by_id(normalized_records)
        relations_by_step = self._relations_by_step(
            normalized_plan,
            normalized_relations,
            records_by_id,
        )
        empty_reasons = self._empty_reasons(
            normalized_plan,
            empty_reason_by_step,
        )
        step_results = tuple(
            self._evaluate_step(
                step,
                relations=relations_by_step[step.step_id],
                records_by_id=records_by_id,
                empty_reason=empty_reasons.get(step.step_id, "no_hits"),
            )
            for step in normalized_plan.steps
        )
        required_ids = {
            step.step_id for step in normalized_plan.steps if step.required
        }
        covered_ids = {
            result.step_id
            for result in step_results
            if result.status == "covered"
        }
        covered_steps = len(covered_ids)
        coverage_ratio = covered_steps / len(normalized_plan.steps)
        covered_required_steps = len(required_ids & covered_ids)
        decision = self._decision(
            normalized_plan,
            step_results=step_results,
            coverage_ratio=coverage_ratio,
            allow_replan=allow_replan,
        )
        return KnowledgePlanCoverage(
            step_results=step_results,
            required_steps=len(required_ids),
            covered_required_steps=covered_required_steps,
            covered_steps=covered_steps,
            coverage_ratio=coverage_ratio,
            replanned=replanned,
            decision=decision,
        )

    @staticmethod
    def _records_by_id(
        records: Sequence[KnowledgeChunkRecord],
    ) -> dict[str, KnowledgeChunkRecord]:
        result: dict[str, KnowledgeChunkRecord] = {}
        for record in records:
            if record.chunk_id in result:
                raise ValueError("覆盖检查快照包含重复 Chunk ID")
            result[record.chunk_id] = record
        return result

    @staticmethod
    def _relations_by_step(
        plan: KnowledgeReasoningPlan,
        relations: Sequence[KnowledgePlanEvidenceRelation],
        records_by_id: Mapping[str, KnowledgeChunkRecord],
    ) -> dict[str, list[KnowledgePlanEvidenceRelation]]:
        plan_step_ids = {step.step_id for step in plan.steps}
        grouped = {step_id: [] for step_id in plan_step_ids}
        seen_pairs: set[tuple[str, str]] = set()
        for relation in relations:
            pair = (relation.step_id, relation.chunk_id)
            if pair in seen_pairs:
                raise ValueError("覆盖检查关系包含重复步骤与 Chunk 对")
            seen_pairs.add(pair)
            if relation.step_id not in plan_step_ids:
                raise ValueError("覆盖检查关系引用未知步骤")
            if relation.chunk_id not in records_by_id:
                raise ValueError("覆盖检查关系引用快照外 Chunk")
            grouped[relation.step_id].append(relation)
        return grouped

    @staticmethod
    def _empty_reasons(
        plan: KnowledgeReasoningPlan,
        values: Mapping[str, KnowledgePlanReasonCode],
    ) -> dict[str, KnowledgePlanReasonCode]:
        if not isinstance(values, Mapping):
            raise ValueError("覆盖检查空结果原因必须是映射")
        plan_step_ids = {step.step_id for step in plan.steps}
        result: dict[str, KnowledgePlanReasonCode] = {}
        for step_id, reason_code in values.items():
            if step_id not in plan_step_ids:
                raise ValueError("覆盖检查空结果原因引用未知步骤")
            if reason_code not in _EMPTY_REASON_CODES:
                raise ValueError("覆盖检查空结果原因码不安全")
            result[step_id] = reason_code
        return result

    @classmethod
    def _evaluate_step(
        cls,
        step: KnowledgePlanStep,
        *,
        relations: Sequence[KnowledgePlanEvidenceRelation],
        records_by_id: Mapping[str, KnowledgeChunkRecord],
        empty_reason: KnowledgePlanReasonCode,
    ) -> KnowledgePlanStepResult:
        if empty_reason == "search_failed":
            return KnowledgePlanStepResult(
                step_id=step.step_id,
                status="failed",
                search_query=step.query,
                reason_code="search_failed",
            )
        trusted_relations = tuple(
            sorted(
                (
                    relation
                    for relation in relations
                    if relation.support_level != "none"
                ),
                key=lambda relation: (-relation.score, relation.chunk_id),
            )[:6]
        )
        direct_relations = tuple(
            relation
            for relation in trusted_relations
            if relation.support_level == "direct"
            and cls._record_covers_subjects(
                records_by_id[relation.chunk_id],
                step.target_subjects,
            )
        )
        selected_relations = direct_relations or trusted_relations
        selected_chunk_ids = tuple(
            relation.chunk_id for relation in selected_relations
        )
        selected_document_ids = cls._unique_document_ids(
            selected_chunk_ids,
            records_by_id,
        )
        if direct_relations:
            return KnowledgePlanStepResult(
                step_id=step.step_id,
                status="covered",
                search_query=step.query,
                selected_chunk_ids=selected_chunk_ids,
                selected_document_ids=selected_document_ids,
                reason_code="enough_evidence",
            )
        if trusted_relations:
            return KnowledgePlanStepResult(
                step_id=step.step_id,
                status="weak",
                search_query=step.query,
                selected_chunk_ids=selected_chunk_ids,
                selected_document_ids=selected_document_ids,
                reason_code="insufficient_subject_coverage",
            )
        uncovered_reason = (
            empty_reason
            if empty_reason in {"no_hits", "scope_filtered", "stale_candidates"}
            else "no_hits"
        )
        return KnowledgePlanStepResult(
            step_id=step.step_id,
            status="uncovered",
            search_query=step.query,
            reason_code=uncovered_reason,
        )

    @staticmethod
    def _unique_document_ids(
        chunk_ids: Sequence[str],
        records_by_id: Mapping[str, KnowledgeChunkRecord],
    ) -> tuple[str, ...]:
        document_ids: list[str] = []
        seen: set[str] = set()
        for chunk_id in chunk_ids:
            document_id = records_by_id[chunk_id].document_id
            if document_id not in seen:
                document_ids.append(document_id)
                seen.add(document_id)
        return tuple(document_ids)

    @classmethod
    def _record_covers_subjects(
        cls,
        record: KnowledgeChunkRecord,
        target_subjects: Sequence[str],
    ) -> bool:
        normalized_record = cls._normalized_text(
            " ".join(
                (
                    record.title,
                    *record.heading_path,
                    record.content,
                )
            )
        )
        return all(
            cls._normalized_text(subject) in normalized_record
            for subject in target_subjects
        )

    @staticmethod
    def _normalized_text(value: str) -> str:
        return " ".join(value.split()).casefold()

    @classmethod
    def _decision(
        cls,
        plan: KnowledgeReasoningPlan,
        *,
        step_results: Sequence[KnowledgePlanStepResult],
        coverage_ratio: float,
        allow_replan: bool,
    ) -> str:
        status_by_id = {
            result.step_id: result.status for result in step_results
        }
        required_covered = all(
            not step.required or status_by_id[step.step_id] == "covered"
            for step in plan.steps
        )
        final_sufficient = cls._final_sufficient(
            plan,
            status_by_id=status_by_id,
            coverage_ratio=coverage_ratio,
            required_covered=required_covered,
        )
        first_round_insufficient = not final_sufficient
        if plan.question_type == "analytical":
            first_round_insufficient = (
                not required_covered or coverage_ratio < 0.75
            )
        if allow_replan and first_round_insufficient:
            return "replan"
        return "answer" if final_sufficient else "insufficient_evidence"

    @staticmethod
    def _final_sufficient(
        plan: KnowledgeReasoningPlan,
        *,
        status_by_id: Mapping[str, str],
        coverage_ratio: float,
        required_covered: bool,
    ) -> bool:
        if not required_covered:
            return False
        if plan.question_type == "analytical":
            return any(
                step.required
                and step.facet in {"subject", "definition"}
                and status_by_id[step.step_id] == "covered"
                for step in plan.steps
            )
        if plan.question_type == "exploratory":
            return coverage_ratio >= 0.75
        subject_steps = [
            step
            for step in plan.steps
            if step.required and len(step.target_subjects) == 1
        ]
        covered_subjects = {
            step.target_subjects[0].casefold()
            for step in subject_steps
            if status_by_id[step.step_id] == "covered"
        }
        has_common_dimension = any(
            step.required
            and step.facet == "comparison"
            and status_by_id[step.step_id] == "covered"
            and len(
                {
                    subject.casefold() for subject in step.target_subjects
                }
                & covered_subjects
            )
            >= 2
            for step in plan.steps
        )
        return len(covered_subjects) >= 2 and has_common_dimension


__all__ = ["KnowledgePlanCoverageChecker"]
