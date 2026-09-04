"""知识问答离线评估基线的定向自验证。"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from app.domain.services.knowledge_document_chunker import KnowledgeDocumentChunker
from app.infrastructure.database.sqlite.knowledge_repository import (
    SQLiteKnowledgeRepository,
)
from app.infrastructure.retrieval.knowledge_search import InMemoryKnowledgeSearch
from app.models.document import Document
from app.models.knowledge_qa import KnowledgeChunkRecord, KnowledgeSearchResult
from assess.knowledge_qa_evaluation import (
    KnowledgeQaEvaluator,
    evaluate_knowledge_qa,
    load_knowledge_qa_cases,
)
from assess.metrics import (
    answerability_accuracy,
    false_support_rate,
    required_fact_coverage,
)
from assess.models import (
    EvidenceRequirement,
    KnowledgeQaEvaluationCase,
    MetricResult,
    SuccessSummary,
    ViolationSummary,
)


class KnowledgeQaEvaluationModelTests(unittest.TestCase):
    """验证知识问答评估模型和纯指标的边界。"""

    def test_answerable_case_requires_documents_and_facts(self) -> None:
        case = KnowledgeQaEvaluationCase(
            case_id="event-loop",
            question="事件循环负责什么？",
            answerable=True,
            expected_document_ids=("doc-python",),
            required_evidence=(
                EvidenceRequirement(
                    fact_id="scheduling",
                    any_of=("事件循环负责调度协程",),
                ),
            ),
            expected_action="answer",
            expected_reason_code="enough_evidence",
            tags=frozenset(("direct_fact",)),
        )
        self.assertEqual(case.expected_document_ids, ("doc-python",))

        with self.assertRaises(ValueError):
            KnowledgeQaEvaluationCase(
                case_id="invalid",
                question="无标注事实",
                answerable=True,
                expected_action="answer",
                expected_reason_code="enough_evidence",
            )

    def test_unanswerable_case_rejects_expected_facts(self) -> None:
        with self.assertRaises(ValueError):
            KnowledgeQaEvaluationCase(
                case_id="invalid-negative",
                question="Python GIL 如何工作？",
                answerable=False,
                expected_document_ids=("doc-python",),
                required_evidence=(
                    EvidenceRequirement("gil", ("GIL",)),
                ),
                expected_action="refuse",
                expected_reason_code="no_relevant_evidence",
            )

    def test_action_annotations_enforce_action_specific_payloads(self) -> None:
        rewrite = KnowledgeQaEvaluationCase(
            case_id="rewrite",
            question="What workloads best fit Project Loom?",
            answerable=True,
            expected_document_ids=("doc-java",),
            required_evidence=(
                EvidenceRequirement("fit", ("阻塞式任务",)),
            ),
            expected_action="rewrite",
            expected_reason_code="low_relevance_retry_available",
            expected_rewrite_terms=("Java", "虚拟线程"),
        )
        self.assertEqual(rewrite.expected_action, "rewrite")

        invalid_cases = (
            dict(
                expected_action="rewrite",
                expected_reason_code="low_relevance_retry_available",
            ),
            dict(
                expected_action="select",
                expected_reason_code="multiple_skill_candidates",
                expected_option_ids=("only-one",),
            ),
            dict(
                expected_action="answer",
                expected_reason_code="no_relevant_evidence",
            ),
        )
        for action_fields in invalid_cases:
            with self.subTest(action_fields=action_fields):
                with self.assertRaises(ValueError):
                    KnowledgeQaEvaluationCase(
                        case_id="invalid-action",
                        question="动作标注无效",
                        answerable=False,
                        **action_fields,
                    )

    def test_fact_and_answerability_metrics_use_explicit_denominators(self) -> None:
        coverage = required_fact_coverage(((True, False), (True,), ()))
        accuracy = answerability_accuracy(
            (True, False, True),
            (True, True, False),
        )
        false_support = false_support_rate(
            (True, False, False),
            (True, True, False),
        )

        self.assertEqual(coverage, MetricResult(0.75, 2, 1))
        self.assertEqual(accuracy, SuccessSummary(1 / 3, 1, 3))
        self.assertEqual(false_support, ViolationSummary(0.5, 1, 2))


class KnowledgeQaCaseLoaderTests(unittest.TestCase):
    """验证磁盘标注加载、规范化和严格拒绝规则。"""

    def setUp(self) -> None:
        """为每个验证创建隔离的临时目录。"""

        self._temporary_directory = tempfile.TemporaryDirectory()
        self._directory = Path(self._temporary_directory.name)
        self._file_index = 0

    def tearDown(self) -> None:
        """释放临时标注目录。"""

        self._temporary_directory.cleanup()

    def _write_json(self, payload: object) -> Path:
        """写入一份 UTF-8 临时 JSON，并返回可供加载的路径。"""

        self._file_index += 1
        path = self._directory / f"cases-{self._file_index}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def test_loader_normalizes_valid_case(self) -> None:
        path = self._write_json(
            [
                {
                    "case_id": "event-loop",
                    "question": " 事件循环负责什么？ ",
                    "answerable": True,
                    "expected_document_ids": ["doc-python"],
                    "required_evidence": [
                        {
                            "fact_id": "scheduling",
                            "any_of": ["事件循环负责调度协程"],
                        }
                    ],
                    "document_scope": [],
                    "expected_action": "answer",
                    "expected_reason_code": "enough_evidence",
                    "expected_rewrite_terms": [],
                    "expected_option_ids": [],
                    "tags": ["direct_fact"],
                }
            ]
        )

        cases = load_knowledge_qa_cases(path)

        self.assertEqual(cases[0].question, "事件循环负责什么？")
        self.assertEqual(cases[0].tags, frozenset(("direct_fact",)))
        self.assertEqual(cases[0].expected_action, "answer")

    def test_loader_rejects_unknown_fields_duplicate_ids_and_invalid_negative(
        self,
    ) -> None:
        invalid_payloads = (
            [{"case_id": "x", "question": "q", "answerable": False, "extra": 1}],
            [
                {"case_id": "x", "question": "q1", "answerable": False},
                {"case_id": "x", "question": "q2", "answerable": False},
            ],
            [
                {
                    "case_id": "negative",
                    "question": "q",
                    "answerable": False,
                    "expected_document_ids": ["doc"],
                    "expected_action": "refuse",
                    "expected_reason_code": "no_relevant_evidence",
                }
            ],
            [
                {
                    "case_id": "invalid-select",
                    "question": "q",
                    "answerable": False,
                    "expected_action": "select",
                    "expected_reason_code": "multiple_skill_candidates",
                    "expected_option_ids": ["only-one"],
                }
            ],
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    load_knowledge_qa_cases(self._write_json(payload))


class _FailFirstSearch:
    """第一次检索失败，后续请求委托真实 BM25。"""

    def __init__(self) -> None:
        self._delegate = InMemoryKnowledgeSearch()
        self._should_fail = True

    async def refresh(self, chunks: Sequence[KnowledgeChunkRecord]) -> None:
        await self._delegate.refresh(chunks)

    async def search(
        self,
        question: str,
        *,
        limit: int = 5,
        document_ids: Sequence[str] = (),
    ) -> KnowledgeSearchResult:
        if self._should_fail:
            self._should_fail = False
            raise RuntimeError("fixture failure")
        return await self._delegate.search(
            question,
            limit=limit,
            document_ids=document_ids,
        )

    async def aclose(self) -> None:
        await self._delegate.aclose()


class KnowledgeQaEvaluatorTests(unittest.IsolatedAsyncioTestCase):
    """验证评估器复用真实组件、事实保护和单题失败隔离。"""

    def setUp(self) -> None:
        """建立包含两个文档的临时 SQLite 与固定标注。"""

        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        directory = Path(self.temporary_directory.name)
        self.database_path = directory / "knowledge.sqlite3"
        self.repository = SQLiteKnowledgeRepository(self.database_path)
        now = datetime(2026, 8, 10, tzinfo=timezone.utc)
        fixtures = (
            (
                "doc-python",
                "Python 异步编程",
                "# Python\n\n## 事件循环\n\n事件循环负责调度协程。",
            ),
            (
                "doc-java",
                "Java 并发编程",
                "# Java\n\n## 线程池\n\n线程池负责复用工作线程。",
            ),
        )
        for document_id, title, content in fixtures:
            chunks = KnowledgeDocumentChunker().chunk(document_id, content)
            self.repository.replace_document(
                Document(
                    document_id=document_id,
                    title=title,
                    content_markdown=content,
                    topics=[title.split()[0]],
                    content_type="tutorial",
                    difficulty="intermediate",
                    author_id="author-fixture",
                    content_hash=hashlib.sha256(
                        content.encode("utf-8")
                    ).hexdigest(),
                    created_at=now,
                    updated_at=now,
                ),
                chunks,
            )

        self.case_path = directory / "positive-cases.json"
        self.case_path.write_text(
            json.dumps(
                [
                    {
                        "case_id": "event-loop",
                        "question": "事件循环负责什么？",
                        "answerable": True,
                        "expected_document_ids": ["doc-python"],
                        "required_evidence": [
                            {
                                "fact_id": "scheduling",
                                "any_of": ["事件循环负责调度协程"],
                            }
                        ],
                        "document_scope": [],
                        "expected_action": "answer",
                        "expected_reason_code": "enough_evidence",
                        "expected_rewrite_terms": [],
                        "expected_option_ids": [],
                        "tags": ["direct_fact"],
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.negative_case_path = directory / "negative-cases.json"
        self.negative_case_path.write_text(
            json.dumps(
                [
                    {
                        "case_id": "unknown-redis",
                        "question": "Redis RDB 和 AOF 有什么区别？",
                        "answerable": False,
                        "expected_document_ids": [],
                        "required_evidence": [],
                        "document_scope": [],
                        "expected_action": "refuse",
                        "expected_reason_code": "no_relevant_evidence",
                        "expected_rewrite_terms": [],
                        "expected_option_ids": [],
                        "tags": ["unanswerable"],
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.cases = (
            KnowledgeQaEvaluationCase(
                case_id="event-loop",
                question="事件循环负责什么？",
                answerable=True,
                expected_document_ids=("doc-python",),
                required_evidence=(
                    EvidenceRequirement(
                        fact_id="scheduling",
                        any_of=("事件循环负责调度协程",),
                    ),
                ),
                expected_action="answer",
                expected_reason_code="enough_evidence",
                tags=frozenset(("direct_fact",)),
            ),
            KnowledgeQaEvaluationCase(
                case_id="thread-pool",
                question="线程池负责什么？",
                answerable=True,
                expected_document_ids=("doc-java",),
                required_evidence=(
                    EvidenceRequirement(
                        fact_id="reuse",
                        any_of=("线程池负责复用工作线程",),
                    ),
                ),
                expected_action="answer",
                expected_reason_code="enough_evidence",
                tags=frozenset(("direct_fact",)),
            ),
        )

    async def test_evaluator_uses_bm25_evidence_and_degraded_citations(
        self,
    ) -> None:
        report = await evaluate_knowledge_qa(
            self.case_path,
            repository_path=self.database_path,
            k=2,
        )

        case = report.cases[0]
        self.assertEqual(case.execution_status, "executed")
        self.assertEqual(case.ranked_document_ids[0], "doc-python")
        self.assertEqual(case.matched_fact_ids, ("scheduling",))
        self.assertTrue(case.predicted_answerable)
        self.assertTrue(case.citation_integrity)
        self.assertEqual(case.vector_status, "skipped")
        self.assertEqual(case.predicted_action, "answer")
        self.assertEqual(case.predicted_reason_code, "enough_evidence")
        self.assertEqual(case.rewrite_count, 0)
        self.assertFalse(case.non_answer_evidence_leak)
        self.assertIsNotNone(report.action_metrics)
        assert report.action_metrics is not None
        self.assertEqual(report.action_metrics.macro_f1, 1.0)
        self.assertFalse(report.used_external_services)

    async def test_unanswerable_case_counts_false_support_without_external_model(
        self,
    ) -> None:
        report = await evaluate_knowledge_qa(
            self.negative_case_path,
            repository_path=self.database_path,
            k=2,
        )

        self.assertEqual(report.total_cases, 1)
        self.assertEqual(report.cases[0].answer_status, "insufficient_evidence")
        self.assertFalse(report.cases[0].predicted_answerable)

    async def test_case_failure_is_recorded_and_later_case_continues(self) -> None:
        evaluator = KnowledgeQaEvaluator(
            repository=self.repository,
            search=_FailFirstSearch(),
        )
        self.addAsyncCleanup(evaluator.aclose)

        report = await evaluator.evaluate(self.cases, k=2)

        self.assertEqual(report.failed_cases, 1)
        self.assertEqual(report.executed_cases, 1)
        self.assertEqual(report.cases[0].execution_status, "failed")
        self.assertEqual(report.cases[0].error_code, "runtime_error")
        self.assertEqual(report.cases[1].execution_status, "executed")


class RealKnowledgeQaCasesTests(unittest.TestCase):
    """验证仓库固定样本的规模、顺序与覆盖类别。"""

    def test_fixed_cases_have_approved_size_and_categories(self) -> None:
        cases = load_knowledge_qa_cases()

        self.assertEqual(len(cases), 35)
        self.assertEqual(cases[0].case_id, "virtual_thread_best_fit")
        self.assertEqual(cases[-1].case_id, "select_runtime_skill")
        self.assertEqual(sum(not case.answerable for case in cases), 8)
        self.assertEqual(
            {case.expected_action for case in cases},
            {"answer", "rewrite", "ask", "select", "refuse"},
        )
        self.assertEqual(
            {tag for case in cases for tag in case.tags},
            {
                "direct_fact",
                "multi_chunk",
                "lexical_mismatch",
                "scoped",
                "cross_document",
                "unanswerable",
                "action_routing",
            },
        )


class RealKnowledgeQaEvaluationTests(unittest.IsolatedAsyncioTestCase):
    """验证真实 SQLite 固定基线完整执行且结果可重复。"""

    async def test_real_baseline_executes_all_cases_without_external_services(
        self,
    ) -> None:
        report = await evaluate_knowledge_qa(k=5)

        self.assertEqual(report.total_cases, 35)
        self.assertEqual(report.executed_cases, 35)
        self.assertEqual(report.failed_cases, 0)
        self.assertEqual(report.not_run_cases, 0)
        self.assertFalse(report.used_external_services)
        self.assertTrue(
            all(case.vector_status == "skipped" for case in report.cases)
        )
        self.assertIsNotNone(report.overall)
        self.assertIsNotNone(report.latency)
        assert report.latency is not None
        self.assertEqual(report.latency.sample_count, 35)
        self.assertIsNotNone(report.action_metrics)
        assert report.action_metrics is not None
        self.assertEqual(report.action_metrics.non_answer_evidence_leak_count, 0)

    async def test_real_baseline_rankings_and_findings_are_repeatable(self) -> None:
        first = await evaluate_knowledge_qa(k=5)
        second = await evaluate_knowledge_qa(k=5)

        self.assertIsNotNone(first.overall)
        self.assertIsNotNone(second.overall)
        self.assertEqual(
            [case.ranked_document_ids for case in first.cases],
            [case.ranked_document_ids for case in second.cases],
        )
        self.assertEqual(first.findings, second.findings)
        self.assertEqual(first.overall, second.overall)
