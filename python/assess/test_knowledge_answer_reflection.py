"""知识回答自动反思的严格模型、策略、Agent 与接线探针。"""

from __future__ import annotations

import importlib.util
import importlib
import inspect
import json
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from assess import test_retrieval_components as retrieval_fixtures

from app.agents.knowledge_answer_reflection_agent import (
    KnowledgeAnswerReflectionAgent,
)
from app.agents.knowledge_answer_agent import KnowledgeAnswerAgent
from app.application.knowledge_answer_reflection import (
    KnowledgeAnswerReflectionService,
)
from app.application.knowledge_qa import KnowledgeQaService
from app.models.evidence_routing import EvidenceOption, KnowledgeEvidenceDecision
from app.models.knowledge_qa import (
    KnowledgeChunkRecord,
    KnowledgeGeneratedAnswer,
    KnowledgeQueryAnalysis,
    KnowledgeSearchHit,
    KnowledgeSearchResult,
)
from app.models.knowledge_reflection import (
    KnowledgeAnswerReflectionAnalysis,
    KnowledgeAnswerReflectionDecision,
    KnowledgeAnswerRevisionPolicy,
    KnowledgeAnswerRiskSignals,
)
from app.domain.services.knowledge_answer_reflection_policy import (
    KnowledgeAnswerReflectionPolicy,
)


def _chunk(
    chunk_id: str,
    *,
    document_id: str = "doc-1",
    title: str = "文档一",
    content: str = "虚拟线程由 JVM 调度，并适合大量阻塞式 I/O 任务。",
) -> KnowledgeChunkRecord:
    """构造不依赖真实数据库的固定知识 Chunk。"""

    return KnowledgeChunkRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        title=title,
        topics=["Java"],
        content_type="technical_design",
        difficulty="intermediate",
        author_id="author-1",
        position=0,
        heading_path=("虚拟线程",),
        content=content,
        content_hash="a" * 64,
        token_count=40,
    )


class _FakeReflectionLlm:
    """捕获结构化反思消息的最小异步 Fake。"""

    def __init__(self, result: object = None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.call_count = 0
        self.messages: list[object] = []
        self.closed = False

    async def ainvoke(self, messages: object) -> object:
        self.call_count += 1
        self.messages = list(messages)  # type: ignore[arg-type]
        if self.error is not None:
            raise self.error
        return self.result

    async def aclose(self) -> None:
        self.closed = True


class _CountingReflectionAgent:
    """记录协调器调用次数的结构化反思 Fake。"""

    def __init__(
        self,
        result: KnowledgeAnswerReflectionAnalysis | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result or KnowledgeAnswerReflectionAnalysis(
            action="answer",
            issue="none",
            confidence=1.0,
        )
        self.error = error
        self.call_count = 0
        self.closed = False

    async def review(self, **_: object) -> KnowledgeAnswerReflectionAnalysis:
        self.call_count += 1
        if self.error is not None:
            raise self.error
        return self.result.model_copy(deep=True)

    async def aclose(self) -> None:
        self.closed = True


class _CapturingAnswerLlm:
    """返回固定 Claim 并捕获 AnswerAgent 输入。"""

    def __init__(self, output: object) -> None:
        self.output = output
        self.messages: list[object] = []

    async def ainvoke(self, messages: object) -> object:
        self.messages = list(messages)  # type: ignore[arg-type]
        return self.output


class _ReflectionKnowledgeRepository:
    """提供简单知识问答接线所需的最小同步仓储。"""

    def __init__(self, records: tuple[KnowledgeChunkRecord, ...]) -> None:
        self.records = records

    def list_ready_chunks(self) -> tuple[KnowledgeChunkRecord, ...]:
        return self.records

    def get_chunks_by_ids(
        self,
        chunk_ids: tuple[str, ...],
    ) -> tuple[KnowledgeChunkRecord, ...]:
        records_by_id = {record.chunk_id: record for record in self.records}
        return tuple(
            records_by_id[chunk_id]
            for chunk_id in chunk_ids
            if chunk_id in records_by_id
        )

    def list_ready_images_by_chunk_ids(
        self,
        chunk_ids: tuple[str, ...],
    ) -> tuple[()]:
        del chunk_ids
        return ()


class _ReflectionKnowledgeSearch:
    """返回固定命中的简单检索 Fake。"""

    def __init__(
        self,
        record: KnowledgeChunkRecord,
        *,
        scores: tuple[float, ...] = (1.0,),
    ) -> None:
        self.record = record
        self.scores = scores
        self.calls: list[str] = []

    async def refresh(
        self,
        chunks: tuple[KnowledgeChunkRecord, ...],
    ) -> None:
        del chunks

    async def search(
        self,
        question: str,
        *,
        limit: int = 5,
        document_ids: tuple[str, ...] = (),
    ) -> KnowledgeSearchResult:
        del limit, document_ids
        self.calls.append(question)
        score = self.scores[min(len(self.calls) - 1, len(self.scores) - 1)]
        return KnowledgeSearchResult(
            hits=(
                KnowledgeSearchHit(
                    chunk_id=self.record.chunk_id,
                    content_hash=self.record.content_hash,
                    score=score,
                    bm25_rank=1,
                ),
            )
        )

    async def aclose(self) -> None:
        return None


class _ReflectionQueryAnalysisAgent:
    """返回固定 direct 查询分析。"""

    def __init__(
        self,
        *,
        retry_query: str | None = None,
        question_type: str = "factual",
        strategy: str = "direct",
        sub_queries: tuple[str, ...] = (),
    ) -> None:
        self.retry_query = retry_query
        self.question_type = question_type
        self.strategy = strategy
        self.sub_queries = sub_queries

    async def analyze(self, question: str, **_: object) -> KnowledgeQueryAnalysis:
        return KnowledgeQueryAnalysis(
            standalone_query=question,
            question_type=self.question_type,
            strategy=self.strategy,
            sub_queries=self.sub_queries,
            retry_query=self.retry_query,
            confidence=1.0,
        )

    async def aclose(self) -> None:
        return None


class _SequencedKnowledgeAnswerAgent:
    """按顺序返回草稿并记录修订策略。"""

    def __init__(self, *answers: KnowledgeGeneratedAnswer) -> None:
        self.answers = answers
        self.calls: list[dict[str, object]] = []

    async def generate(self, **kwargs: object) -> KnowledgeGeneratedAnswer:
        self.calls.append(dict(kwargs))
        index = min(len(self.calls) - 1, len(self.answers) - 1)
        return self.answers[index].model_copy(deep=True)

    async def aclose(self) -> None:
        return None


class _FakeReflectionService:
    """返回固定五类决策并记录反思与修复复检。"""

    def __init__(
        self,
        decision: KnowledgeAnswerReflectionDecision,
        *,
        repaired_decision: KnowledgeAnswerReflectionDecision | None = None,
    ) -> None:
        self.decision = decision
        self.repaired_decision = repaired_decision or (
            KnowledgeAnswerReflectionDecision(
                action="answer",
                confidence=1.0,
                reason_code="repair_validation_pass",
                approved=True,
            )
        )
        self.calls: list[dict[str, object]] = []
        self.repaired_calls: list[dict[str, object]] = []
        self.close_calls = 0

    async def review(
        self,
        **kwargs: object,
    ) -> KnowledgeAnswerReflectionDecision:
        self.calls.append(dict(kwargs))
        return self.decision.model_copy(deep=True)

    def validate_repaired(
        self,
        **kwargs: object,
    ) -> KnowledgeAnswerReflectionDecision:
        self.repaired_calls.append(dict(kwargs))
        return self.repaired_decision.model_copy(deep=True)

    async def aclose(self) -> None:
        self.close_calls += 1


class _RewriteThenAnswerEvidenceGate:
    """首次要求查询改写、第二次批准证据的门控 Fake。"""

    def __init__(self, retry_query: str) -> None:
        self.retry_query = retry_query
        self.after_retrieval_calls = 0

    def precheck(self, **_: object) -> None:
        return None

    def decide_after_retrieval(
        self,
        signals: object,
        **_: object,
    ) -> KnowledgeEvidenceDecision:
        self.after_retrieval_calls += 1
        if self.after_retrieval_calls == 1:
            return KnowledgeEvidenceDecision(
                action="rewrite",
                confidence=1.0,
                reason_code="low_relevance_retry_available",
                rewritten_query=self.retry_query,
            )
        return KnowledgeEvidenceDecision(
            action="answer",
            confidence=1.0,
            reason_code="enough_evidence",
            approved_evidence_ids=tuple(getattr(signals, "selected_evidence_ids")),
        )


class KnowledgeReflectionModuleTests(unittest.TestCase):
    """先固定新组件模块边界，再逐步补齐行为探针。"""

    def test_reflection_model_module_exists(self) -> None:
        """批准的严格模型必须位于独立领域契约模块。"""

        self.assertIsNotNone(
            importlib.util.find_spec("app.models.knowledge_reflection")
        )

    def test_reflection_model_exports_required_contracts(self) -> None:
        """模型模块必须提供计划中批准的四个请求期契约。"""

        module = importlib.import_module("app.models.knowledge_reflection")
        required = {
            "KnowledgeAnswerReflectionAnalysis",
            "KnowledgeAnswerReflectionDecision",
            "KnowledgeAnswerRevisionPolicy",
            "KnowledgeAnswerRiskSignals",
        }
        self.assertEqual(
            {name for name in required if hasattr(module, name)},
            required,
        )


class KnowledgeReflectionModelTests(unittest.TestCase):
    """固定反思动作字段组合与有界输入。"""

    def test_reflection_decision_requires_action_specific_payload(self) -> None:
        """改写必须给出模式，回答不能夹带改写字段。"""

        decision = KnowledgeAnswerReflectionDecision(
            action="rewrite",
            confidence=0.9,
            reason_code="incomplete_answer",
            rewrite_mode="regenerate_answer",
            revision_policy=KnowledgeAnswerRevisionPolicy(
                focus=("coverage",),
            ),
        )
        self.assertEqual(decision.rewrite_mode, "regenerate_answer")

        with self.assertRaises(ValidationError):
            KnowledgeAnswerReflectionDecision(
                action="answer",
                confidence=1.0,
                reason_code="semantic_pass",
                approved=True,
                rewrite_mode="regenerate_answer",
                revision_policy=KnowledgeAnswerRevisionPolicy(
                    focus=("coverage",),
                ),
            )

    def test_non_answer_decision_cannot_approve_draft(self) -> None:
        """问、选、拒和改都不能把草稿标记为已批准。"""

        with self.assertRaises(ValidationError):
            KnowledgeAnswerReflectionDecision(
                action="refuse",
                confidence=1.0,
                reason_code="invalid_citation",
                approved=True,
            )

    def test_retry_retrieval_requires_protected_query(self) -> None:
        """重新检索必须携带上游提供的有界查询。"""

        with self.assertRaises(ValidationError):
            KnowledgeAnswerReflectionDecision(
                action="rewrite",
                confidence=0.8,
                reason_code="unsupported_claim",
                rewrite_mode="retry_retrieval",
                revision_policy=KnowledgeAnswerRevisionPolicy(
                    focus=("grounding",),
                ),
            )

    def test_ask_and_select_require_safe_payload(self) -> None:
        """询问需要问题，选择只能携带二到五个可信选项。"""

        ask = KnowledgeAnswerReflectionDecision(
            action="ask",
            confidence=0.8,
            reason_code="missing_information",
            clarification_question="请补充需要比较的第二个对象。",
        )
        self.assertEqual(ask.action, "ask")

        select = KnowledgeAnswerReflectionDecision(
            action="select",
            confidence=0.8,
            reason_code="ambiguous_target",
            options=(
                EvidenceOption(option_id="doc-1", label="文档一"),
                EvidenceOption(option_id="doc-2", label="文档二"),
            ),
        )
        self.assertEqual(len(select.options), 2)

        with self.assertRaises(ValidationError):
            KnowledgeAnswerReflectionDecision(
                action="select",
                confidence=0.8,
                reason_code="ambiguous_target",
                options=(EvidenceOption(option_id="doc-1", label="文档一"),),
            )

    def test_analysis_rejects_contradictory_action_and_issue(self) -> None:
        """模型候选不能声称通过同时报告答案问题。"""

        with self.assertRaises(ValidationError):
            KnowledgeAnswerReflectionAnalysis(
                action="answer",
                issue="unsupported_claim",
                confidence=0.9,
            )

    def test_revision_policy_rejects_duplicates_and_unknown_focus(self) -> None:
        """修订策略只允许非重复的枚举关注点。"""

        with self.assertRaises(ValidationError):
            KnowledgeAnswerRevisionPolicy(focus=("coverage", "coverage"))
        with self.assertRaises(ValidationError):
            KnowledgeAnswerRevisionPolicy(focus=("free_text",))

    def test_risk_signals_reject_duplicate_or_empty_evidence_ids(self) -> None:
        """风险信号必须携带唯一、非空的批准 Chunk 身份。"""

        with self.assertRaises(ValidationError):
            KnowledgeAnswerRiskSignals(
                question_type="factual",
                approved_chunk_ids=("chunk-1", "chunk-1"),
                cited_chunk_ids=("chunk-1",),
                document_count=1,
            )


class KnowledgeReflectionPolicyModuleTests(unittest.TestCase):
    """固定确定性 Policy 的独立模块边界。"""

    def test_reflection_policy_module_exists(self) -> None:
        self.assertIsNotNone(
            importlib.util.find_spec(
                "app.domain.services.knowledge_answer_reflection_policy"
            )
        )

    def test_reflection_policy_module_exports_policy(self) -> None:
        module = importlib.import_module(
            "app.domain.services.knowledge_answer_reflection_policy"
        )
        self.assertTrue(hasattr(module, "KnowledgeAnswerReflectionPolicy"))

    def test_reflection_policy_exposes_three_protected_operations(self) -> None:
        policy = KnowledgeAnswerReflectionPolicy()
        self.assertTrue(callable(getattr(policy, "precheck", None)))
        self.assertTrue(callable(getattr(policy, "protect", None)))
        self.assertTrue(callable(getattr(policy, "fallback", None)))


class KnowledgeAnswerReflectionPolicyTests(unittest.TestCase):
    """固定硬错误、低风险直通和语义候选保护。"""

    def setUp(self) -> None:
        self.policy = KnowledgeAnswerReflectionPolicy()

    @staticmethod
    def signals(**updates: object) -> KnowledgeAnswerRiskSignals:
        payload: dict[str, object] = {
            "question_type": "factual",
            "approved_chunk_ids": ("chunk-1",),
            "approved_image_ids": (),
            "cited_chunk_ids": ("chunk-1",),
            "cited_image_ids": (),
            "document_count": 1,
        }
        payload.update(updates)
        return KnowledgeAnswerRiskSignals.model_validate(payload)

    def test_low_risk_single_document_answer_passes_without_agent(self) -> None:
        decision = self.policy.precheck(self.signals())
        assert decision is not None
        self.assertEqual(decision.action, "answer")
        self.assertEqual(decision.reason_code, "deterministic_pass")
        self.assertTrue(decision.approved)

    def test_invalid_citation_is_refused_before_semantic_review(self) -> None:
        decision = self.policy.precheck(self.signals(cited_chunk_ids=("unknown",)))
        assert decision is not None
        self.assertEqual(decision.action, "refuse")
        self.assertEqual(decision.reason_code, "invalid_citation")

    def test_answer_agent_degradation_is_refused_without_semantic_review(self) -> None:
        decision = self.policy.precheck(self.signals(answer_degraded=True))
        assert decision is not None
        self.assertEqual(decision.action, "refuse")
        self.assertEqual(decision.reason_code, "generation_unavailable")

    def test_multi_document_or_complex_answer_requires_semantic_review(self) -> None:
        self.assertIsNone(self.policy.precheck(self.signals(document_count=2)))
        self.assertIsNone(
            self.policy.precheck(self.signals(question_type="comparative"))
        )
        self.assertIsNone(
            self.policy.precheck(self.signals(force_semantic_review=True))
        )

    def test_semantic_pass_approves_current_draft(self) -> None:
        decision = self.policy.protect(
            KnowledgeAnswerReflectionAnalysis(
                action="answer",
                issue="none",
                confidence=0.9,
            ),
            signals=self.signals(document_count=2),
        )
        self.assertEqual(decision.action, "answer")
        self.assertEqual(decision.reason_code, "semantic_pass")
        self.assertTrue(decision.approved)

    def test_regenerate_candidate_uses_enum_revision_policy(self) -> None:
        decision = self.policy.protect(
            KnowledgeAnswerReflectionAnalysis(
                action="rewrite",
                issue="incomplete_answer",
                confidence=0.9,
                rewrite_mode="regenerate_answer",
                revision_focus=("coverage",),
            ),
            signals=self.signals(document_count=2),
        )
        self.assertEqual(decision.action, "rewrite")
        self.assertEqual(decision.rewrite_mode, "regenerate_answer")
        assert decision.revision_policy is not None
        self.assertEqual(decision.revision_policy.focus, ("coverage",))

    def test_retry_retrieval_requires_unused_protected_query(self) -> None:
        analysis = KnowledgeAnswerReflectionAnalysis(
            action="rewrite",
            issue="unsupported_claim",
            confidence=0.9,
            rewrite_mode="retry_retrieval",
            revision_focus=("grounding",),
        )
        refused = self.policy.protect(
            analysis,
            signals=self.signals(document_count=2),
        )
        self.assertEqual(refused.action, "refuse")
        self.assertEqual(refused.reason_code, "retry_query_unavailable")

        rewritten = self.policy.protect(
            analysis,
            signals=self.signals(document_count=2),
            retry_query="受保护的重试查询",
        )
        self.assertEqual(rewritten.action, "rewrite")
        self.assertEqual(rewritten.rewritten_query, "受保护的重试查询")

    def test_rewrite_after_repair_budget_is_refused(self) -> None:
        decision = self.policy.protect(
            KnowledgeAnswerReflectionAnalysis(
                action="rewrite",
                issue="off_topic",
                confidence=0.8,
                rewrite_mode="regenerate_answer",
                revision_focus=("relevance",),
            ),
            signals=self.signals(
                document_count=2,
                repair_attempted=True,
            ),
        )
        self.assertEqual(decision.action, "refuse")
        self.assertEqual(decision.reason_code, "repair_exhausted")

    def test_ambiguous_target_uses_only_trusted_options(self) -> None:
        analysis = KnowledgeAnswerReflectionAnalysis(
            action="select",
            issue="ambiguous_target",
            confidence=0.8,
        )
        selected = self.policy.protect(
            analysis,
            signals=self.signals(document_count=2),
            trusted_options=(
                EvidenceOption(option_id="doc-1", label="文档一"),
                EvidenceOption(option_id="doc-2", label="文档二"),
            ),
        )
        self.assertEqual(selected.action, "select")
        self.assertEqual(len(selected.options), 2)

        asked = self.policy.protect(
            analysis,
            signals=self.signals(document_count=2),
        )
        self.assertEqual(asked.action, "ask")
        self.assertEqual(asked.reason_code, "ambiguous_target")

    def test_fallback_preserves_valid_draft_and_marks_degradation(self) -> None:
        decision = self.policy.fallback(self.signals(document_count=2))
        self.assertEqual(decision.action, "answer")
        self.assertEqual(
            decision.reason_code,
            "reflection_unavailable_fallback",
        )
        self.assertTrue(decision.reflection_degraded)


class KnowledgeReflectionAgentModuleTests(unittest.TestCase):
    """固定结构化反思 Agent 的独立模块边界。"""

    def test_reflection_agent_module_exists(self) -> None:
        self.assertIsNotNone(
            importlib.util.find_spec("app.agents.knowledge_answer_reflection_agent")
        )

    def test_reflection_agent_module_exports_agent(self) -> None:
        module = importlib.import_module("app.agents.knowledge_answer_reflection_agent")
        self.assertTrue(hasattr(module, "KnowledgeAnswerReflectionAgent"))


class KnowledgeAnswerReflectionAgentTests(unittest.IsolatedAsyncioTestCase):
    """验证 Agent 只接收有界白名单数据并返回严格候选。"""

    async def test_agent_uses_bounded_evidence_and_returns_schema(self) -> None:
        llm = _FakeReflectionLlm(
            {
                "decision": {
                    "kind": "rewrite",
                    "issue": "incomplete_answer",
                    "rewrite_mode": "regenerate_answer",
                    "revision_focus": ["coverage"],
                },
                "confidence": 0.9,
            }
        )
        agent = KnowledgeAnswerReflectionAgent(llm=llm)

        result = await agent.review(
            question="比较两种方案",
            standalone_query="比较方案 A 与方案 B",
            answer="当前草稿只介绍了方案 A。",
            evidence=(
                _chunk("chunk-a"),
                _chunk(
                    "chunk-b",
                    document_id="doc-2",
                    title="文档二",
                    content="方案 B 使用不同的调度机制。",
                ),
            ),
            question_type="comparative",
        )

        self.assertEqual(result.issue, "incomplete_answer")
        self.assertEqual(llm.call_count, 1)
        self.assertEqual(len(llm.messages), 2)
        envelope = json.loads(str(llm.messages[1].content))
        self.assertEqual(set(envelope), {"contract", "input"})
        self.assertEqual(
            envelope["contract"]["name"],
            "knowledge_answer_reflection",
        )
        self.assertEqual(envelope["contract"]["version"], 2)
        schema = envelope["contract"]["output_schema"]
        self.assertIn("decision", schema["properties"])
        human_payload = envelope["input"]
        self.assertEqual(
            set(human_payload),
            {
                "question",
                "standalone_query",
                "question_type",
                "answer",
                "evidence",
                "images",
                "coverage",
            },
        )
        self.assertNotIn("runtime_skill", human_payload)
        self.assertNotIn("user_id", human_payload)

    async def test_agent_rejects_empty_evidence_before_model_call(self) -> None:
        llm = _FakeReflectionLlm(
            KnowledgeAnswerReflectionAnalysis(
                action="answer",
                issue="none",
                confidence=1.0,
            )
        )
        agent = KnowledgeAnswerReflectionAgent(llm=llm)

        with self.assertRaises(ValueError):
            await agent.review(
                question="问题",
                standalone_query="问题",
                answer="草稿",
                evidence=(),
                question_type="factual",
            )

        self.assertEqual(llm.call_count, 0)

    async def test_agent_rejects_invalid_model_output(self) -> None:
        agent = KnowledgeAnswerReflectionAgent(
            llm=_FakeReflectionLlm(
                {
                    "action": "answer",
                    "issue": "unsupported_claim",
                    "confidence": 0.9,
                }
            )
        )

        with self.assertRaises(ValidationError):
            await agent.review(
                question="问题",
                standalone_query="问题",
                answer="草稿",
                evidence=(_chunk("chunk-1"),),
                question_type="factual",
            )

    async def test_agent_without_llm_fails_safely_and_closes_client(self) -> None:
        agent = KnowledgeAnswerReflectionAgent(llm=None)
        with self.assertRaises(RuntimeError):
            await agent.review(
                question="问题",
                standalone_query="问题",
                answer="草稿",
                evidence=(_chunk("chunk-1"),),
                question_type="factual",
            )

        llm = _FakeReflectionLlm()
        owned = KnowledgeAnswerReflectionAgent(llm=llm)
        await owned.aclose()
        self.assertTrue(llm.closed)

    async def test_agent_propagates_model_failure_for_service_fallback(self) -> None:
        agent = KnowledgeAnswerReflectionAgent(
            llm=_FakeReflectionLlm(error=TimeoutError("timeout"))
        )
        with self.assertRaises(TimeoutError):
            await agent.review(
                question="问题",
                standalone_query="问题",
                answer="草稿",
                evidence=(_chunk("chunk-1"),),
                question_type="factual",
            )


class KnowledgeReflectionServiceModuleTests(unittest.TestCase):
    """固定应用层协调器的独立模块边界。"""

    def test_reflection_service_module_exists(self) -> None:
        self.assertIsNotNone(
            importlib.util.find_spec("app.application.knowledge_answer_reflection")
        )

    def test_reflection_service_module_exports_service(self) -> None:
        module = importlib.import_module("app.application.knowledge_answer_reflection")
        self.assertTrue(hasattr(module, "KnowledgeAnswerReflectionService"))


class KnowledgeAnswerReflectionServiceTests(unittest.IsolatedAsyncioTestCase):
    """固定风险触发、单次调用、fallback 和修复复检。"""

    @staticmethod
    def answer(*, cited_chunk_ids: tuple[str, ...] = ("chunk-1",)):
        return KnowledgeGeneratedAnswer(
            answer="虚拟线程由 JVM 调度。",
            cited_chunk_ids=cited_chunk_ids,
        )

    @staticmethod
    def service(
        agent: _CountingReflectionAgent | None,
    ) -> KnowledgeAnswerReflectionService:
        return KnowledgeAnswerReflectionService(
            policy=KnowledgeAnswerReflectionPolicy(),
            agent=agent,
        )

    async def test_low_risk_answer_does_not_call_agent(self) -> None:
        agent = _CountingReflectionAgent()
        service = self.service(agent)

        decision = await service.review(
            question="什么是虚拟线程？",
            standalone_query="什么是虚拟线程？",
            question_type="factual",
            answer=self.answer(),
            evidence=(_chunk("chunk-1"),),
        )

        self.assertEqual(decision.action, "answer")
        self.assertEqual(decision.reason_code, "deterministic_pass")
        self.assertEqual(agent.call_count, 0)

    async def test_semantic_risk_calls_agent_once(self) -> None:
        agent = _CountingReflectionAgent()
        service = self.service(agent)

        decision = await service.review(
            question="比较两种方案",
            standalone_query="比较方案 A 和方案 B",
            question_type="comparative",
            answer=self.answer(cited_chunk_ids=("chunk-1", "chunk-2")),
            evidence=(
                _chunk("chunk-1"),
                _chunk("chunk-2", document_id="doc-2", title="文档二"),
            ),
        )

        self.assertEqual(decision.action, "answer")
        self.assertEqual(decision.reason_code, "semantic_pass")
        self.assertEqual(agent.call_count, 1)

    async def test_hard_failure_does_not_call_agent(self) -> None:
        agent = _CountingReflectionAgent()
        service = self.service(agent)

        decision = await service.review(
            question="问题",
            standalone_query="问题",
            question_type="comparative",
            answer=self.answer(cited_chunk_ids=("unknown",)),
            evidence=(_chunk("chunk-1"),),
        )

        self.assertEqual(decision.action, "refuse")
        self.assertEqual(decision.reason_code, "invalid_citation")
        self.assertEqual(agent.call_count, 0)

    async def test_agent_failure_uses_deterministic_fallback(self) -> None:
        agent = _CountingReflectionAgent(error=TimeoutError("timeout"))
        service = self.service(agent)

        decision = await service.review(
            question="比较两种方案",
            standalone_query="比较方案 A 和方案 B",
            question_type="comparative",
            answer=self.answer(cited_chunk_ids=("chunk-1", "chunk-2")),
            evidence=(
                _chunk("chunk-1"),
                _chunk("chunk-2", document_id="doc-2", title="文档二"),
            ),
        )

        self.assertEqual(decision.action, "answer")
        self.assertEqual(
            decision.reason_code,
            "reflection_unavailable_fallback",
        )
        self.assertTrue(decision.reflection_degraded)
        self.assertEqual(agent.call_count, 1)

    async def test_missing_agent_uses_same_fallback(self) -> None:
        service = self.service(None)
        decision = await service.review(
            question="比较两种方案",
            standalone_query="比较方案 A 和方案 B",
            question_type="comparative",
            answer=self.answer(cited_chunk_ids=("chunk-1", "chunk-2")),
            evidence=(
                _chunk("chunk-1"),
                _chunk("chunk-2", document_id="doc-2", title="文档二"),
            ),
        )
        self.assertEqual(
            decision.reason_code,
            "reflection_unavailable_fallback",
        )

    async def test_repaired_answer_uses_only_deterministic_validation(self) -> None:
        agent = _CountingReflectionAgent()
        service = self.service(agent)

        passed = service.validate_repaired(
            question_type="comparative",
            answer=self.answer(cited_chunk_ids=("chunk-1", "chunk-2")),
            evidence=(
                _chunk("chunk-1"),
                _chunk("chunk-2", document_id="doc-2", title="文档二"),
            ),
        )
        refused = service.validate_repaired(
            question_type="factual",
            answer=self.answer(cited_chunk_ids=("unknown",)),
            evidence=(_chunk("chunk-1"),),
        )

        self.assertEqual(passed.reason_code, "repair_validation_pass")
        self.assertEqual(refused.reason_code, "repair_exhausted")
        self.assertEqual(agent.call_count, 0)

    async def test_service_closes_owned_agent_once(self) -> None:
        agent = _CountingReflectionAgent()
        service = self.service(agent)
        await service.aclose()
        self.assertTrue(agent.closed)


class KnowledgeAnswerRevisionTests(unittest.IsolatedAsyncioTestCase):
    """验证 AnswerAgent 只接收枚举化修订策略。"""

    async def test_answer_agent_accepts_only_enum_revision_policy(self) -> None:
        llm = _CapturingAnswerLlm(
            {
                "outcome": "answer",
                "claims": [
                    {
                        "text": "虚拟线程由 JVM 调度。",
                        "evidence_ids": ["chunk-1"],
                        "image_ids": [],
                    }
                ]
            }
        )
        agent = KnowledgeAnswerAgent(llm=llm)
        policy = KnowledgeAnswerRevisionPolicy(
            focus=("grounding", "coverage"),
        )

        generated = await agent.generate(
            question="问题",
            evidence=(_chunk("chunk-1"),),
            revision_policy=policy,
        )

        self.assertFalse(generated.degraded)
        payload = json.loads(str(llm.messages[1].content))["input"]
        self.assertEqual(
            payload["revision_policy"],
            {"focus": ["grounding", "coverage"]},
        )
        self.assertIn("不能增加", str(llm.messages[0].content))

    async def test_invalid_revision_policy_falls_back_to_no_revision(self) -> None:
        llm = _CapturingAnswerLlm(
            {
                "outcome": "answer",
                "claims": [
                    {
                        "text": "虚拟线程由 JVM 调度。",
                        "evidence_ids": ["chunk-1"],
                        "image_ids": [],
                    }
                ]
            }
        )
        agent = KnowledgeAnswerAgent(llm=llm)

        await agent.generate(
            question="问题",
            evidence=(_chunk("chunk-1"),),
            revision_policy=object(),  # type: ignore[arg-type]
        )

        payload = json.loads(str(llm.messages[1].content))["input"]
        self.assertIsNone(payload["revision_policy"])

    def test_knowledge_answer_generator_protocol_exposes_revision_policy(
        self,
    ) -> None:
        module = importlib.import_module("app.application.knowledge_qa")
        signature = inspect.signature(module.KnowledgeAnswerGenerator.generate)
        self.assertIn("revision_policy", signature.parameters)


class KnowledgeQaReflectionContractTests(unittest.TestCase):
    """固定知识问答应用服务的可替换反思边界。"""

    def test_knowledge_service_accepts_optional_reflection_service(self) -> None:
        module = importlib.import_module("app.application.knowledge_qa")
        signature = inspect.signature(module.KnowledgeQaService.__init__)
        self.assertIn("reflection_service", signature.parameters)

    def test_reflection_events_expose_only_safe_decision_metadata(self) -> None:
        """反思事件只能记录动作、原因和预算状态，不能记录草稿正文。"""

        source = inspect.getsource(KnowledgeQaService._review_generated_answer)

        self.assertIn('stage="答案反思"', source)
        self.assertIn('"reason_code": decision.reason_code', source)
        self.assertNotIn("generated.answer", source)

        repair_source = inspect.getsource(KnowledgeQaService._regenerate_answer_once)
        self.assertIn('stage="答案修复"', repair_source)
        self.assertNotIn("generated.answer", repair_source)


class KnowledgeQaReflectionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """验证简单知识链在提交答案前执行一次反思与有界修复。"""

    @staticmethod
    def _service(
        *,
        reflection: _FakeReflectionService,
        answer_agent: _SequencedKnowledgeAnswerAgent | None = None,
        retry_query: str | None = None,
        search: _ReflectionKnowledgeSearch | None = None,
        evidence_gate: object | None = None,
    ) -> KnowledgeQaService:
        record = _chunk("chunk-1")
        return KnowledgeQaService(
            repository=_ReflectionKnowledgeRepository((record,)),  # type: ignore[arg-type]
            search=search or _ReflectionKnowledgeSearch(record),
            answer_agent=answer_agent
            or _SequencedKnowledgeAnswerAgent(
                KnowledgeGeneratedAnswer(
                    answer="虚拟线程由 JVM 调度。",
                    cited_chunk_ids=(record.chunk_id,),
                )
            ),
            query_analysis_agent=_ReflectionQueryAnalysisAgent(retry_query=retry_query),
            evidence_gate=evidence_gate,  # type: ignore[arg-type]
            reflection_service=reflection,
        )

    async def test_simple_answer_is_reviewed_before_finalize(self) -> None:
        """简单答案必须经过反思服务后才能组装公开引用。"""

        reflection = _FakeReflectionService(
            KnowledgeAnswerReflectionDecision(
                action="answer",
                confidence=1.0,
                reason_code="deterministic_pass",
                approved=True,
            )
        )
        service = self._service(reflection=reflection)

        result = await service.ask("什么是虚拟线程？", document_ids=("doc-1",))

        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.citations), 1)
        self.assertEqual(len(reflection.calls), 1)

    async def test_simple_abstain_skips_reflection_and_returns_no_citations(
        self,
    ) -> None:
        """主动拒答必须在反思前短路，不能制造引用。"""

        reflection = _FakeReflectionService(
            KnowledgeAnswerReflectionDecision(
                action="answer",
                confidence=1.0,
                reason_code="deterministic_pass",
                approved=True,
            )
        )
        answer_agent = _SequencedKnowledgeAnswerAgent(
            KnowledgeGeneratedAnswer(
                outcome="abstain",
                answer="当前证据不足。",
                abstain_reason="insufficient_evidence",
            )
        )
        service = self._service(
            reflection=reflection,
            answer_agent=answer_agent,
        )

        result = await service.ask("什么是虚拟线程？", document_ids=("doc-1",))

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.answer, "当前证据不足，无法可靠回答该问题。")
        self.assertEqual(result.citations, ())
        self.assertEqual(result.images, ())
        self.assertEqual(len(reflection.calls), 0)

    async def test_regenerate_answer_runs_once_then_validates_deterministically(
        self,
    ) -> None:
        """同证据改写只再生成一次，修复后不递归调用反思模型。"""

        reflection = _FakeReflectionService(
            KnowledgeAnswerReflectionDecision(
                action="rewrite",
                confidence=0.9,
                reason_code="incomplete_answer",
                rewrite_mode="regenerate_answer",
                revision_policy=KnowledgeAnswerRevisionPolicy(
                    focus=("coverage",),
                ),
            )
        )
        answer_agent = _SequencedKnowledgeAnswerAgent(
            KnowledgeGeneratedAnswer(
                answer="初版只覆盖部分内容。",
                cited_chunk_ids=("chunk-1",),
            ),
            KnowledgeGeneratedAnswer(
                answer="修订版完整覆盖证据已有内容。",
                cited_chunk_ids=("chunk-1",),
            ),
        )
        service = self._service(
            reflection=reflection,
            answer_agent=answer_agent,
        )

        result = await service.ask("什么是虚拟线程？", document_ids=("doc-1",))

        self.assertEqual(result.status, "success")
        self.assertEqual(result.answer, "修订版完整覆盖证据已有内容。")
        self.assertEqual(len(answer_agent.calls), 2)
        self.assertEqual(len(reflection.calls), 1)
        self.assertEqual(len(reflection.repaired_calls), 1)
        revision = answer_agent.calls[1]["revision_policy"]
        self.assertEqual(
            revision,
            KnowledgeAnswerRevisionPolicy(focus=("coverage",)),
        )

    async def test_non_answer_decisions_clear_public_evidence(self) -> None:
        """问、选、拒必须短路答案引用和图片组装。"""

        decisions = (
            KnowledgeAnswerReflectionDecision(
                action="ask",
                confidence=0.9,
                reason_code="missing_information",
                clarification_question="请补充目标 Java 版本。",
            ),
            KnowledgeAnswerReflectionDecision(
                action="select",
                confidence=0.9,
                reason_code="ambiguous_target",
                options=(
                    EvidenceOption(option_id="doc-1", label="文档一"),
                    EvidenceOption(option_id="doc-2", label="文档二"),
                ),
            ),
            KnowledgeAnswerReflectionDecision(
                action="refuse",
                confidence=1.0,
                reason_code="unsupported_claim",
            ),
        )
        expected_statuses = (
            "needs_clarification",
            "needs_clarification",
            "insufficient_evidence",
        )

        for decision, expected_status in zip(
            decisions,
            expected_statuses,
            strict=True,
        ):
            with self.subTest(action=decision.action):
                service = self._service(reflection=_FakeReflectionService(decision))

                result = await service.ask(
                    "什么是虚拟线程？",
                    document_ids=("doc-1",),
                )

                self.assertEqual(result.status, expected_status)
                self.assertEqual(result.citations, ())
                self.assertEqual(result.images, ())
                assert result.execution_trace is not None
                self.assertEqual(result.execution_trace.documents, ())

    async def test_invalid_repaired_answer_is_refused_without_citations(
        self,
    ) -> None:
        """自动修复未通过硬边界复检时不能提交修订草稿。"""

        reflection = _FakeReflectionService(
            KnowledgeAnswerReflectionDecision(
                action="rewrite",
                confidence=0.9,
                reason_code="incomplete_answer",
                rewrite_mode="regenerate_answer",
                revision_policy=KnowledgeAnswerRevisionPolicy(
                    focus=("coverage",),
                ),
            ),
            repaired_decision=KnowledgeAnswerReflectionDecision(
                action="refuse",
                confidence=1.0,
                reason_code="repair_exhausted",
            ),
        )
        answer_agent = _SequencedKnowledgeAnswerAgent(
            KnowledgeGeneratedAnswer(
                answer="初版。",
                cited_chunk_ids=("chunk-1",),
            ),
            KnowledgeGeneratedAnswer(
                answer="仍然无效的修订版。",
                cited_chunk_ids=("unknown",),
            ),
        )
        service = self._service(
            reflection=reflection,
            answer_agent=answer_agent,
        )

        result = await service.ask("什么是虚拟线程？", document_ids=("doc-1",))

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.citations, ())
        self.assertEqual(result.images, ())
        self.assertEqual(len(reflection.calls), 1)
        self.assertEqual(len(reflection.repaired_calls), 1)

    async def test_retry_retrieval_uses_unused_retry_query_once(self) -> None:
        """检索改写只使用上游受保护查询，并在修复后确定性复检。"""

        retry_query = "虚拟线程 JVM 调度 阻塞 I/O"
        reflection = _FakeReflectionService(
            KnowledgeAnswerReflectionDecision(
                action="rewrite",
                confidence=0.9,
                reason_code="unsupported_claim",
                rewrite_mode="retry_retrieval",
                rewritten_query=retry_query,
                revision_policy=KnowledgeAnswerRevisionPolicy(
                    focus=("grounding",),
                ),
            )
        )
        record = _chunk("chunk-1")
        search = _ReflectionKnowledgeSearch(record)
        answer_agent = _SequencedKnowledgeAnswerAgent(
            KnowledgeGeneratedAnswer(
                answer="初版。",
                cited_chunk_ids=("chunk-1",),
            ),
            KnowledgeGeneratedAnswer(
                answer="重新检索后的修订版。",
                cited_chunk_ids=("chunk-1",),
            ),
        )
        service = self._service(
            reflection=reflection,
            answer_agent=answer_agent,
            retry_query=retry_query,
            search=search,
        )

        result = await service.ask("什么是虚拟线程？", document_ids=("doc-1",))

        self.assertEqual(result.status, "success")
        self.assertEqual(result.answer, "重新检索后的修订版。")
        self.assertEqual(search.calls, ["什么是虚拟线程？", retry_query])
        self.assertEqual(len(answer_agent.calls), 2)
        self.assertEqual(len(reflection.calls), 1)
        self.assertEqual(len(reflection.repaired_calls), 1)

    async def test_retry_retrieval_is_blocked_after_evidence_gate_rewrite(
        self,
    ) -> None:
        """证据门已消费查询改写预算时，反思不得启动第三次检索。"""

        retry_query = "虚拟线程 JVM 调度 阻塞 I/O"
        reflection = _FakeReflectionService(
            KnowledgeAnswerReflectionDecision(
                action="rewrite",
                confidence=0.9,
                reason_code="unsupported_claim",
                rewrite_mode="retry_retrieval",
                rewritten_query=retry_query,
                revision_policy=KnowledgeAnswerRevisionPolicy(
                    focus=("grounding",),
                ),
            )
        )
        record = _chunk("chunk-1")
        search = _ReflectionKnowledgeSearch(
            record,
            scores=(0.0, 1.0),
        )
        service = self._service(
            reflection=reflection,
            retry_query=retry_query,
            search=search,
            evidence_gate=_RewriteThenAnswerEvidenceGate(retry_query),
        )

        result = await service.ask("什么是虚拟线程？", document_ids=("doc-1",))

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(search.calls, ["什么是虚拟线程？", retry_query])
        self.assertTrue(reflection.calls[0]["query_rewrite_attempted"])


class KnowledgeFeedbackReflectionTests(unittest.IsolatedAsyncioTestCase):
    """验证原可信证据再回答复用反思门且不获得第二次补救预算。"""

    async def test_feedback_regeneration_forces_semantic_review(self) -> None:
        """反馈再回答必须强制检查并标记已经处于修复阶段。"""

        record = _chunk("chunk-1")
        reflection = _FakeReflectionService(
            KnowledgeAnswerReflectionDecision(
                action="answer",
                confidence=0.9,
                reason_code="semantic_pass",
                approved=True,
            )
        )
        service = KnowledgeQaService(
            repository=_ReflectionKnowledgeRepository((record,)),  # type: ignore[arg-type]
            answer_agent=_SequencedKnowledgeAnswerAgent(
                KnowledgeGeneratedAnswer(
                    answer="按反馈修正后的回答。",
                    cited_chunk_ids=(record.chunk_id,),
                )
            ),
            reflection_service=reflection,
        )

        result = await service.regenerate_from_evidence(
            "请先给结论",
            chunk_ids=(record.chunk_id,),
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(len(reflection.calls), 1)
        self.assertTrue(reflection.calls[0]["force_semantic_review"])
        self.assertTrue(reflection.calls[0]["repair_attempted"])
        self.assertFalse(reflection.calls[0]["allow_retrieval_retry"])

    async def test_feedback_regeneration_abstain_skips_reflection(self) -> None:
        """原证据再回答主动拒答时不得进入反馈反思。"""

        record = _chunk("chunk-1")
        reflection = _FakeReflectionService(
            KnowledgeAnswerReflectionDecision(
                action="answer",
                confidence=1.0,
                reason_code="deterministic_pass",
                approved=True,
            )
        )
        service = KnowledgeQaService(
            repository=_ReflectionKnowledgeRepository((record,)),  # type: ignore[arg-type]
            answer_agent=_SequencedKnowledgeAnswerAgent(
                KnowledgeGeneratedAnswer(
                    outcome="abstain",
                    answer="当前证据不足。",
                    abstain_reason="conflicting_evidence",
                )
            ),
            reflection_service=reflection,
        )

        result = await service.regenerate_from_evidence(
            "请先给结论",
            chunk_ids=(record.chunk_id,),
        )

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.answer, "当前证据不足，无法可靠回答该问题。")
        self.assertEqual(result.citations, ())
        self.assertEqual(len(reflection.calls), 0)

    async def test_feedback_regeneration_cannot_start_second_repair(self) -> None:
        """反馈补救后的 rewrite 候选必须按预算耗尽拒绝。"""

        record = _chunk("chunk-1")
        reflection = _FakeReflectionService(
            KnowledgeAnswerReflectionDecision(
                action="rewrite",
                confidence=0.9,
                reason_code="incomplete_answer",
                rewrite_mode="regenerate_answer",
                revision_policy=KnowledgeAnswerRevisionPolicy(
                    focus=("coverage",),
                ),
            )
        )
        answer_agent = _SequencedKnowledgeAnswerAgent(
            KnowledgeGeneratedAnswer(
                answer="仍然不完整的反馈修订。",
                cited_chunk_ids=(record.chunk_id,),
            )
        )
        service = KnowledgeQaService(
            repository=_ReflectionKnowledgeRepository((record,)),  # type: ignore[arg-type]
            answer_agent=answer_agent,
            reflection_service=reflection,
        )

        result = await service.regenerate_from_evidence(
            "请补充遗漏内容",
            chunk_ids=(record.chunk_id,),
        )

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.citations, ())
        self.assertEqual(len(answer_agent.calls), 1)
        self.assertEqual(len(reflection.calls), 1)


class KnowledgeQaComplexReflectionTests(unittest.IsolatedAsyncioTestCase):
    """验证复杂规划答案只允许同证据修订，不启动第二轮规划。"""

    @staticmethod
    def _service(
        *,
        reflection: _FakeReflectionService,
        answer_agent: _SequencedKnowledgeAnswerAgent,
    ) -> tuple[
        KnowledgeQaService,
        retrieval_fixtures._FixedKnowledgeReasoningPlanner,
    ]:
        plan = retrieval_fixtures.KnowledgeQaServiceTests._complex_plan_fixture(
            "comparative"
        )
        records = tuple(_chunk(f"chunk-{step.step_id}") for step in plan.steps)
        outcome = retrieval_fixtures.KnowledgeQaServiceTests._round_outcome_fixture(
            plan,
            records,
            support_by_step={
                step.step_id: (
                    f"chunk-{step.step_id}",
                    "direct",
                    1.0,
                )
                for step in plan.steps
            },
        )
        coverage = retrieval_fixtures.KnowledgeQaServiceTests._coverage_fixture(
            plan,
            {step.step_id: "covered" for step in plan.steps},
            decision="answer",
        )
        planner = retrieval_fixtures._FixedKnowledgeReasoningPlanner(plans=[plan])
        service = KnowledgeQaService(
            repository=_ReflectionKnowledgeRepository(records),  # type: ignore[arg-type]
            answer_agent=answer_agent,
            query_analysis_agent=_ReflectionQueryAnalysisAgent(
                question_type="comparative",
                strategy="decomposed",
                sub_queries=("BM25", "向量检索"),
            ),
            reasoning_planner_agent=planner,
            plan_executor=retrieval_fixtures._FixedKnowledgePlanExecutor([outcome]),
            plan_coverage_checker=(
                retrieval_fixtures._SequenceKnowledgePlanCoverageChecker([coverage])
            ),
            reflection_service=reflection,
        )
        return service, planner

    async def test_complex_answer_can_regenerate_once(self) -> None:
        """复杂答案允许基于相同 Coverage 和 Evidence 修订一次。"""

        reflection = _FakeReflectionService(
            KnowledgeAnswerReflectionDecision(
                action="rewrite",
                confidence=0.9,
                reason_code="incomplete_answer",
                rewrite_mode="regenerate_answer",
                revision_policy=KnowledgeAnswerRevisionPolicy(
                    focus=("coverage", "organization"),
                ),
            )
        )
        answer_agent = _SequencedKnowledgeAnswerAgent(
            KnowledgeGeneratedAnswer(
                answer="初版比较不完整。",
                cited_chunk_ids=(
                    "chunk-step-1",
                    "chunk-step-2",
                    "chunk-step-3",
                ),
            ),
            KnowledgeGeneratedAnswer(
                answer="修订后的完整比较。",
                cited_chunk_ids=(
                    "chunk-step-1",
                    "chunk-step-2",
                    "chunk-step-3",
                ),
            ),
        )
        service, planner = self._service(
            reflection=reflection,
            answer_agent=answer_agent,
        )

        result = await service.ask("比较 BM25 和向量检索", document_ids=("doc-1",))

        self.assertEqual(result.status, "success")
        self.assertIn("修订后的完整比较。", result.answer)
        self.assertEqual(len(answer_agent.calls), 2)
        self.assertEqual(len(reflection.calls), 1)
        self.assertEqual(len(reflection.repaired_calls), 1)
        self.assertFalse(reflection.calls[0]["allow_retrieval_retry"])
        self.assertIsNotNone(reflection.calls[0]["coverage"])
        self.assertEqual(len(planner.plan_calls), 1)
        self.assertEqual(len(planner.replan_calls), 0)

    async def test_complex_answer_cannot_retry_retrieval(self) -> None:
        """复杂反思改写不得再启动检索或规划，只能保守拒绝。"""

        reflection = _FakeReflectionService(
            KnowledgeAnswerReflectionDecision(
                action="rewrite",
                confidence=0.9,
                reason_code="unsupported_claim",
                rewrite_mode="retry_retrieval",
                rewritten_query="新的复杂查询",
                revision_policy=KnowledgeAnswerRevisionPolicy(
                    focus=("grounding",),
                ),
            )
        )
        answer_agent = _SequencedKnowledgeAnswerAgent(
            KnowledgeGeneratedAnswer(
                answer="需要重新检索的初版。",
                cited_chunk_ids=(
                    "chunk-step-1",
                    "chunk-step-2",
                    "chunk-step-3",
                ),
            )
        )
        service, planner = self._service(
            reflection=reflection,
            answer_agent=answer_agent,
        )

        result = await service.ask("比较 BM25 和向量检索", document_ids=("doc-1",))

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(result.citations, ())
        self.assertEqual(len(answer_agent.calls), 1)
        self.assertEqual(len(reflection.calls), 1)
        self.assertEqual(len(planner.plan_calls), 1)
        self.assertEqual(len(planner.replan_calls), 0)


class KnowledgeReflectionBootstrapTests(unittest.IsolatedAsyncioTestCase):
    """验证反思组件的服务生命周期和应用启动装配。"""

    def test_bootstrap_injects_small_model_reflection_service(self) -> None:
        """启动期必须构造反思协调器并注入知识问答服务。"""

        module = importlib.import_module("app.bootstrap")
        source = inspect.getsource(module.lifespan)
        helper = getattr(module, "_create_knowledge_answer_reflection", None)

        self.assertIsNotNone(helper)
        assert helper is not None
        self.assertIn(
            "KnowledgeAnswerReflectionAgent.from_settings(settings)",
            inspect.getsource(helper),
        )
        self.assertIn("reflection_service=reflection_service", source)

    def test_bootstrap_falls_back_when_reflection_agent_creation_fails(
        self,
    ) -> None:
        """反思模型创建失败不得阻止知识问答服务启动。"""

        module = importlib.import_module("app.bootstrap")
        with patch.object(
            KnowledgeAnswerReflectionAgent,
            "from_settings",
            side_effect=RuntimeError("模型不可用"),
        ):
            try:
                service = module._create_knowledge_answer_reflection(object())
            except RuntimeError:
                self.fail("反思模型创建失败不应中断启动")

        self.assertIsInstance(service, KnowledgeAnswerReflectionService)
        self.assertIsNone(service._agent)

    async def test_knowledge_service_closes_reflection_service_once(self) -> None:
        """重复关闭知识服务时不得重复关闭反思 Agent。"""

        record = _chunk("chunk-1")
        reflection = _FakeReflectionService(
            KnowledgeAnswerReflectionDecision(
                action="answer",
                confidence=1.0,
                reason_code="deterministic_pass",
                approved=True,
            )
        )
        service = KnowledgeQaService(
            repository=_ReflectionKnowledgeRepository((record,)),  # type: ignore[arg-type]
            search=_ReflectionKnowledgeSearch(record),
            answer_agent=_SequencedKnowledgeAnswerAgent(
                KnowledgeGeneratedAnswer(
                    answer="答案。",
                    cited_chunk_ids=(record.chunk_id,),
                )
            ),
            query_analysis_agent=_ReflectionQueryAnalysisAgent(),
            reflection_service=reflection,
        )

        await service.aclose()
        await service.aclose()

        self.assertEqual(reflection.close_calls, 1)


class KnowledgeReflectionActionEvaluationTests(unittest.IsolatedAsyncioTestCase):
    """使用固定样本评估五类动作、硬拦截和调用预算。"""

    async def test_fixed_action_table_respects_budgets_and_evidence_boundary(
        self,
    ) -> None:
        """固定样本不得泄漏 Evidence，且反思和修复均不超过一次。"""

        semantic_pass = KnowledgeAnswerReflectionAnalysis(
            action="answer",
            issue="none",
            confidence=0.9,
        )
        regenerate = KnowledgeAnswerReflectionAnalysis(
            action="rewrite",
            issue="incomplete_answer",
            confidence=0.9,
            rewrite_mode="regenerate_answer",
            revision_focus=("coverage",),
        )
        retry = KnowledgeAnswerReflectionAnalysis(
            action="rewrite",
            issue="unsupported_claim",
            confidence=0.9,
            rewrite_mode="retry_retrieval",
            revision_focus=("grounding",),
        )
        ask = KnowledgeAnswerReflectionAnalysis(
            action="ask",
            issue="missing_information",
            confidence=0.8,
            missing_information=("目标版本",),
        )
        select = KnowledgeAnswerReflectionAnalysis(
            action="select",
            issue="ambiguous_target",
            confidence=0.8,
        )
        unsafe = KnowledgeAnswerReflectionAnalysis(
            action="refuse",
            issue="unsafe_answer",
            confidence=0.9,
        )
        cases = (
            {
                "name": "factual-low",
                "type": "factual",
                "action": "answer",
                "calls": 0,
                "low": True,
            },
            {
                "name": "procedural-low",
                "type": "procedural",
                "action": "answer",
                "calls": 0,
                "low": True,
            },
            {
                "name": "summary-low",
                "type": "summarization",
                "action": "answer",
                "calls": 0,
                "low": True,
            },
            {
                "name": "generation-hard",
                "type": "factual",
                "action": "refuse",
                "calls": 0,
                "hard": True,
                "answer": "degraded",
            },
            {
                "name": "unknown-chunk-hard",
                "type": "factual",
                "action": "refuse",
                "calls": 0,
                "hard": True,
                "answer": "unknown",
            },
            {
                "name": "unknown-image-hard",
                "type": "factual",
                "action": "refuse",
                "calls": 0,
                "hard": True,
                "answer": "image",
            },
            {
                "name": "comparative-pass",
                "type": "comparative",
                "action": "answer",
                "calls": 1,
            },
            {
                "name": "analytical-pass",
                "type": "analytical",
                "action": "answer",
                "calls": 1,
            },
            {
                "name": "exploratory-pass",
                "type": "exploratory",
                "action": "answer",
                "calls": 1,
            },
            {
                "name": "verification-pass",
                "type": "verification",
                "action": "answer",
                "calls": 1,
            },
            {
                "name": "multi-doc-pass",
                "type": "factual",
                "action": "answer",
                "calls": 1,
                "multi": True,
            },
            {
                "name": "forced-pass",
                "type": "factual",
                "action": "answer",
                "calls": 1,
                "force": True,
            },
            {
                "name": "regenerate",
                "type": "comparative",
                "action": "rewrite",
                "mode": "regenerate_answer",
                "calls": 1,
                "analysis": regenerate,
            },
            {
                "name": "retry",
                "type": "comparative",
                "action": "rewrite",
                "mode": "retry_retrieval",
                "calls": 1,
                "analysis": retry,
                "retry": "受控重试查询",
            },
            {
                "name": "retry-missing",
                "type": "comparative",
                "action": "refuse",
                "calls": 1,
                "analysis": retry,
            },
            {
                "name": "ask",
                "type": "comparative",
                "action": "ask",
                "calls": 1,
                "analysis": ask,
            },
            {
                "name": "select",
                "type": "comparative",
                "action": "select",
                "calls": 1,
                "analysis": select,
                "options": True,
            },
            {
                "name": "select-fallback",
                "type": "comparative",
                "action": "ask",
                "calls": 1,
                "analysis": select,
            },
            {
                "name": "unsafe",
                "type": "comparative",
                "action": "refuse",
                "calls": 1,
                "analysis": unsafe,
            },
            {
                "name": "agent-fallback",
                "type": "comparative",
                "action": "answer",
                "calls": 1,
                "error": True,
            },
        )
        first = _chunk("chunk-1")
        second = _chunk(
            "chunk-2",
            document_id="doc-2",
            title="文档二",
        )
        call_counts: list[int] = []
        repair_counts: list[int] = []
        triggered = 0
        low_risk_triggered = 0
        low_risk_total = 0
        predicted_hard_blocks = 0
        correct_hard_blocks = 0
        non_answer_total = 0
        leaked_non_answers = 0
        repairs_attempted = 0
        repaired_and_passed = 0

        for case in cases:
            with self.subTest(name=case["name"]):
                records = (first, second) if case.get("multi") else (first,)
                answer_kind = case.get("answer")
                answer = KnowledgeGeneratedAnswer(
                    answer="固定草稿。",
                    cited_chunk_ids=(
                        ("unknown",)
                        if answer_kind == "unknown"
                        else tuple(record.chunk_id for record in records)
                    ),
                    cited_image_ids=("unknown-image",)
                    if answer_kind == "image"
                    else (),
                    degraded=answer_kind == "degraded",
                )
                agent = _CountingReflectionAgent(
                    result=case.get("analysis", semantic_pass),
                    error=RuntimeError("反思不可用") if case.get("error") else None,
                )
                service = KnowledgeAnswerReflectionService(
                    policy=KnowledgeAnswerReflectionPolicy(),
                    agent=agent,
                )
                options = (
                    (
                        EvidenceOption(option_id="doc-1", label="文档一"),
                        EvidenceOption(option_id="doc-2", label="文档二"),
                    )
                    if case.get("options")
                    else ()
                )

                decision = await service.review(
                    question="固定问题",
                    standalone_query="固定独立查询",
                    question_type=case["type"],
                    answer=answer,
                    evidence=records,
                    retry_query=case.get("retry"),
                    force_semantic_review=bool(case.get("force")),
                    trusted_options=options,
                )

                self.assertEqual(decision.action, case["action"])
                self.assertEqual(decision.rewrite_mode, case.get("mode"))
                self.assertEqual(agent.call_count, case["calls"])
                call_counts.append(agent.call_count)
                triggered += int(agent.call_count > 0)
                if case.get("low"):
                    low_risk_total += 1
                    low_risk_triggered += int(agent.call_count > 0)
                if case.get("hard"):
                    predicted_hard_blocks += int(decision.action == "refuse")
                    correct_hard_blocks += int(decision.action == case["action"])
                if decision.action != "answer":
                    non_answer_total += 1
                    leaked_non_answers += int(decision.approved)
                repairs = int(decision.action == "rewrite")
                repair_counts.append(repairs)
                if repairs:
                    repairs_attempted += 1
                    repaired = service.validate_repaired(
                        question_type=case["type"],
                        answer=answer,
                        evidence=records,
                    )
                    repaired_and_passed += int(repaired.action == "answer")

        executed = len(cases)
        metrics = {
            "reflection_trigger_rate": triggered / executed,
            "hard_block_precision": (correct_hard_blocks / predicted_hard_blocks),
            "repair_success_rate": repaired_and_passed / repairs_attempted,
            "unnecessary_reflection_rate": (low_risk_triggered / low_risk_total),
            "non_answer_evidence_leak_rate": (leaked_non_answers / non_answer_total),
            "max_reflection_calls_per_request": max(call_counts),
            "max_repairs_per_request": max(repair_counts),
        }

        self.assertGreater(metrics["reflection_trigger_rate"], 0.0)
        self.assertEqual(metrics["hard_block_precision"], 1.0)
        self.assertEqual(metrics["repair_success_rate"], 1.0)
        self.assertEqual(metrics["unnecessary_reflection_rate"], 0.0)
        self.assertEqual(metrics["non_answer_evidence_leak_rate"], 0.0)
        self.assertEqual(metrics["max_reflection_calls_per_request"], 1)
        self.assertEqual(metrics["max_repairs_per_request"], 1)


if __name__ == "__main__":
    unittest.main()
