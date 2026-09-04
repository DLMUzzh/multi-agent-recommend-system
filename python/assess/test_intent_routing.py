"""统一验证对话推荐与知识问答的规则优先路由和 LLM 调用次数。"""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from app.agents.intent_recognition_agent import IntentRecognitionAgent
from app.application.knowledge_qa import KnowledgeQaService
from app.domain.services.conversation_arbitrator import ConversationArbitrator
from app.infrastructure.database.sqlite.knowledge_repository import (
    SQLiteKnowledgeRepository,
)
from app.models.intent import (
    ArbitrationAction,
    IntentName,
    IntentState,
    RecommendationContext,
    RelationHint,
)
from app.models.knowledge_qa import (
    KnowledgeQueryAnalysis,
    KnowledgeSearchResult,
)
from app.orchestration.conversation_graph import ConversationGraph


_CASES_PATH = Path(__file__).parents[2] / "data" / "intent_evaluation_cases.json"


class _CountingIntentLlm:
    """记录一次结构化调用，并返回固定业务决策。"""

    def __init__(self, output: dict[str, Any]) -> None:
        self.output = output
        self.calls = 0
        self.messages: list[Any] = []

    async def ainvoke(self, messages: list[Any]) -> dict[str, Any]:
        self.calls += 1
        self.messages = list(messages)
        return dict(self.output)


class _RecordingQueryAnalyzer:
    """记录知识问答是否执行统一查询分析。"""

    def __init__(self) -> None:
        self.calls = 0

    async def analyze(self, question: str, **_: Any) -> KnowledgeQueryAnalysis:
        self.calls += 1
        return KnowledgeQueryAnalysis(
            standalone_query=(
                question
                if question.startswith("Spring 事务为什么")
                else f"改写：{question}"
            ),
            uses_history=not question.startswith("Spring 事务为什么"),
            question_type="analytical",
            strategy="direct",
            confidence=0.9,
        )

    async def aclose(self) -> None:
        return None


class _RecordingSearch:
    """记录知识服务最终使用的检索查询。"""

    def __init__(self) -> None:
        self.questions: list[str] = []

    async def refresh(self, *_: Any) -> None:
        return None

    async def search(self, question: str, **_: Any) -> KnowledgeSearchResult:
        self.questions.append(question)
        return KnowledgeSearchResult()

    async def aclose(self) -> None:
        return None


def _load_cases() -> dict[str, dict[str, Any]]:
    payload = json.loads(_CASES_PATH.read_text(encoding="utf-8"))
    return {item["id"]: item for item in payload["cases"]}


def _legacy_context() -> RecommendationContext:
    """用当前持久化格式构造旧上下文，实施后继续验证兼容读取。"""

    return RecommendationContext.model_validate(
        {
            "primary_topics": ["Java 虚拟线程"],
            "topic_weights": {"Java 虚拟线程": 1.0},
            "retrieval_query": "Java 虚拟线程 高并发",
            "size": 5,
            "seen_article_ids": ["doc-java"],
        }
    )


class IntentEvaluationDatasetTests(unittest.IsolatedAsyncioTestCase):
    """验证顶层意图固定样本覆盖常见边界且真实消费规则样本。"""

    def test_dataset_covers_required_routing_boundaries(self) -> None:
        cases = _load_cases()

        self.assertGreaterEqual(len(cases), 20)
        self.assertTrue(
            {
                "greeting_no_action",
                "thanks_no_action",
                "english_knowledge_question",
                "negative_explanation_request",
                "mixed_recommend_and_explain",
                "repeat_without_context",
                "history_pronoun_question",
            }.issubset(cases)
        )

    async def test_new_rule_cases_skip_intent_llm(self) -> None:
        cases = _load_cases()
        for case_id in (
            "greeting_no_action",
            "thanks_no_action",
            "compliment_no_action",
            "explicit_summary_question",
            "explicit_verification_question",
            "repeat_without_context",
        ):
            with self.subTest(case_id=case_id):
                case = cases[case_id]
                llm = _CountingIntentLlm({})

                result = await IntentRecognitionAgent(llm=llm).run(case["message"])

                self.assertEqual(result.intent.value, case["intent"])
                self.assertEqual(result.relation.value, case["relation"])
                self.assertEqual(result.rewritten_query, case["expected_query"])
                self.assertEqual(llm.calls, case["intent_llm_calls"])


class IntentPromptContractTests(unittest.IsolatedAsyncioTestCase):
    """验证意图 Prompt 提供稳定置信度和字段组合标尺。"""

    async def test_prompt_defines_confidence_and_payload_combinations(self) -> None:
        llm = _CountingIntentLlm(
            {
                "intent": "unknown",
                "relation": "unclear",
                "rewritten_query": None,
                "updated_intent": None,
                "confidence": 0.3,
            }
        )

        await IntentRecognitionAgent(llm=llm).run("推荐 Java，再解释第二篇")

        prompt = str(llm.messages[0].content)
        self.assertIn("0.90 到 1.00", prompt)
        self.assertIn("0.60 到 0.89", prompt)
        self.assertIn("0.00 到 0.59", prompt)
        self.assertIn("knowledge_qa", prompt)
        self.assertIn("relation=unclear", prompt)
        self.assertIn("updated_intent", prompt)
        self.assertIn("明确冲突", prompt)
        self.assertIn("unknown 也可以使用高 confidence", prompt)

        envelope = json.loads(str(llm.messages[1].content))
        self.assertEqual(set(envelope), {"contract", "input"})
        self.assertEqual(envelope["contract"]["name"], "intent_recognition")
        self.assertEqual(envelope["contract"]["version"], 2)
        self.assertIsInstance(envelope["contract"]["output_schema"], dict)
        self.assertEqual(
            envelope["input"]["message"],
            "推荐 Java，再解释第二篇",
        )

    async def test_invalid_intent_relation_combination_falls_back(self) -> None:
        llm = _CountingIntentLlm(
            {
                "intent": "knowledge_qa",
                "relation": "repeat",
                "rewritten_query": "Spring 事务为什么会失效？",
                "updated_intent": None,
                "confidence": 0.95,
            }
        )

        result = await IntentRecognitionAgent(llm=llm).run("为什么会失效？")

        self.assertEqual(result.intent, IntentName.UNKNOWN)
        self.assertEqual(result.source.value, "fallback")
        self.assertEqual(result.relation, RelationHint.UNCLEAR)

    async def test_discriminated_provider_decision_maps_to_existing_result(
        self,
    ) -> None:
        llm = _CountingIntentLlm(
            {
                "decision": {
                    "kind": "knowledge_qa",
                    "relation": "new",
                    "rewritten_query": "解释 RRF 的工作原理",
                },
                "confidence": 0.94,
            }
        )

        result = await IntentRecognitionAgent(llm=llm).run(
            "推荐 Java，再解释第二篇"
        )

        self.assertEqual(result.intent, IntentName.KNOWLEDGE_QA)
        self.assertEqual(result.relation, RelationHint.NEW)
        self.assertEqual(result.rewritten_query, "解释 RRF 的工作原理")
        schema = json.loads(str(llm.messages[1].content))["contract"][
            "output_schema"
        ]
        self.assertEqual(set(schema["properties"]), {"decision", "confidence"})


class IntentRoutingContractTests(unittest.TestCase):
    """验证新模型只保留实际路由和召回需要的字段。"""

    def test_recommendation_context_uses_minimal_query_contract(self) -> None:
        self.assertEqual(
            set(RecommendationContext.model_fields),
            {"query", "size", "seen_article_ids", "avoid_seen"},
        )

    def test_legacy_context_restores_query_without_persisting_old_fields(self) -> None:
        context = _legacy_context()

        self.assertTrue(hasattr(context, "query"))
        self.assertEqual(context.query, "Java 虚拟线程 高并发")
        self.assertEqual(
            context.model_dump(),
            {
                "query": "Java 虚拟线程 高并发",
                "size": 5,
                "seen_article_ids": ["doc-java"],
                "avoid_seen": False,
            },
        )

    def test_graph_no_longer_accepts_chat_recommendation_rewriter(self) -> None:
        self.assertNotIn(
            "query_rewrite_agent",
            inspect.signature(ConversationGraph.__init__).parameters,
        )

    def test_knowledge_service_accepts_protected_prepared_query(self) -> None:
        self.assertIn(
            "prepared_query",
            inspect.signature(KnowledgeQaService.ask).parameters,
        )


class RuleFirstIntentRoutingTests(unittest.IsolatedAsyncioTestCase):
    """验证明确推荐与明确问答不调用 LLM，歧义场景最多调用一次。"""

    async def test_clear_recommendation_is_rule_routed_without_llm(self) -> None:
        case = _load_cases()["clear_recommendation"]
        llm = _CountingIntentLlm(
            {
                "intent": "unknown",
                "relation": "unclear",
                "rewritten_query": None,
                "updated_intent": None,
                "confidence": 0.0,
            }
        )

        result = await IntentRecognitionAgent(llm=llm).run(case["message"])

        self.assertEqual(result.intent, IntentName.RECOMMEND_ARTICLES)
        self.assertEqual(result.relation, RelationHint.NEW)
        self.assertEqual(result.rewritten_query, case["expected_query"])
        self.assertEqual(result.resolved_intent.size, case["expected_size"])
        self.assertEqual(llm.calls, case["intent_llm_calls"])

    async def test_clear_knowledge_question_is_rule_routed_without_llm(self) -> None:
        case = _load_cases()["clear_knowledge_question"]
        llm = _CountingIntentLlm(
            {
                "intent": "unknown",
                "relation": "unclear",
                "rewritten_query": None,
                "updated_intent": None,
                "confidence": 0.0,
            }
        )

        result = await IntentRecognitionAgent(llm=llm).run(case["message"])

        self.assertEqual(result.intent, IntentName.KNOWLEDGE_QA)
        self.assertEqual(result.rewritten_query, case["expected_query"])
        self.assertIsNone(result.resolved_intent)
        self.assertEqual(llm.calls, case["intent_llm_calls"])

    async def test_history_dependent_question_uses_one_llm_for_route_and_query(
        self,
    ) -> None:
        case = _load_cases()["history_dependent_question"]
        llm = _CountingIntentLlm(
            {
                "intent": case["intent"],
                "relation": case["relation"],
                "rewritten_query": case["expected_query"],
                "updated_intent": None,
                "confidence": 0.95,
            }
        )

        result = await IntentRecognitionAgent(llm=llm).run(case["message"])

        self.assertEqual(result.intent, IntentName.KNOWLEDGE_QA)
        self.assertEqual(result.rewritten_query, case["expected_query"])
        self.assertEqual(llm.calls, case["intent_llm_calls"])

    async def test_repeat_and_quantity_reuse_previous_query_without_llm(self) -> None:
        cases = _load_cases()
        for case_id in ("repeat_recommendation", "repeat_with_size"):
            with self.subTest(case_id=case_id):
                case = cases[case_id]
                llm = _CountingIntentLlm({})

                result = await IntentRecognitionAgent(llm=llm).run(
                    case["message"],
                    active_context=_legacy_context(),
                )

                self.assertEqual(result.intent, IntentName.RECOMMEND_ARTICLES)
                self.assertEqual(result.relation, RelationHint.REPEAT)
                self.assertTrue(hasattr(result, "rewritten_query"))
                self.assertEqual(result.rewritten_query, case["expected_query"])
                self.assertEqual(result.resolved_intent.size, case["expected_size"])
                self.assertEqual(llm.calls, case["intent_llm_calls"])

    async def test_continue_recommendation_switches_from_qa_without_llm(self) -> None:
        case = _load_cases()["continue_from_knowledge"]
        llm = _CountingIntentLlm({})

        result = await IntentRecognitionAgent(llm=llm).run(
            case["message"],
            active_context=_legacy_context(),
            intent_state=IntentState.KNOWLEDGE_QA,
        )

        self.assertEqual(result.intent, IntentName.RECOMMEND_ARTICLES)
        self.assertTrue(hasattr(result, "rewritten_query"))
        self.assertEqual(result.rewritten_query, case["expected_query"])
        self.assertEqual(llm.calls, case["intent_llm_calls"])

    async def test_time_phrase_remains_in_query_instead_of_becoming_filter(self) -> None:
        case = _load_cases()["recommendation_with_time_phrase"]
        llm = _CountingIntentLlm({})

        result = await IntentRecognitionAgent(llm=llm).run(case["message"])

        self.assertEqual(result.intent, IntentName.RECOMMEND_ARTICLES)
        self.assertEqual(result.rewritten_query, case["expected_query"])
        self.assertFalse(hasattr(result.resolved_intent, "publish_time_after"))
        self.assertEqual(llm.calls, case["intent_llm_calls"])

    async def test_language_only_is_deferred_to_single_llm_not_rule_filter(self) -> None:
        case = _load_cases()["language_only_is_not_filter_rule"]
        llm = _CountingIntentLlm(
            {
                "intent": case["intent"],
                "relation": case["relation"],
                "rewritten_query": case["expected_query"],
                "updated_intent": {"resource_type": "article", "size": 5},
                "confidence": 0.95,
            }
        )

        result = await IntentRecognitionAgent(llm=llm).run(
            case["message"],
            active_context=_legacy_context(),
        )

        self.assertEqual(result.intent, IntentName.RECOMMEND_ARTICLES)
        self.assertTrue(hasattr(result, "rewritten_query"))
        self.assertEqual(result.rewritten_query, case["expected_query"])
        self.assertFalse(hasattr(result.resolved_intent, "language"))
        self.assertEqual(llm.calls, case["intent_llm_calls"])

    async def test_invalid_quantity_and_combined_request_use_single_llm(self) -> None:
        for message in ("推荐 11 篇 Java 文章", "给我 5 篇并说明原因"):
            with self.subTest(message=message):
                llm = _CountingIntentLlm(
                    {
                        "intent": "unknown",
                        "relation": "unclear",
                        "rewritten_query": None,
                        "updated_intent": None,
                        "confidence": 0.95,
                    }
                )

                result = await IntentRecognitionAgent(llm=llm).run(
                    message,
                    active_context=_legacy_context(),
                )

                self.assertEqual(result.intent, IntentName.UNKNOWN)
                self.assertEqual(result.source.value, "llm")
                self.assertEqual(llm.calls, 1)

    async def test_mixed_business_and_negative_conflict_follow_llm_decision(
        self,
    ) -> None:
        cases = _load_cases()
        for case_id in ("mixed_recommend_and_explain", "negative_conflict"):
            with self.subTest(case_id=case_id):
                case = cases[case_id]
                llm = _CountingIntentLlm(
                    {
                        "intent": case["intent"],
                        "relation": case["relation"],
                        "rewritten_query": case["expected_query"],
                        "updated_intent": None,
                        "confidence": 0.95,
                    }
                )

                result = await IntentRecognitionAgent(llm=llm).run(
                    case["message"],
                    active_context=_legacy_context(),
                )

                self.assertEqual(result.intent.value, case["intent"])
                self.assertEqual(result.rewritten_query, case["expected_query"])
                self.assertEqual(llm.calls, case["intent_llm_calls"])

    async def test_llm_cannot_add_document_scope_or_overlong_query(self) -> None:
        outputs = (
            {
                "intent": "knowledge_qa",
                "relation": "new",
                "rewritten_query": "Spring 事务为什么会失效？",
                "updated_intent": None,
                "confidence": 0.95,
                "document_ids": ["forged-document"],
            },
            {
                "intent": "knowledge_qa",
                "relation": "new",
                "rewritten_query": "问" * 501,
                "updated_intent": None,
                "confidence": 0.95,
            },
        )
        for output in outputs:
            with self.subTest(output_keys=tuple(output)):
                llm = _CountingIntentLlm(output)

                result = await IntentRecognitionAgent(llm=llm).run(
                    "为什么会失效？"
                )

                self.assertEqual(result.intent, IntentName.UNKNOWN)
                self.assertEqual(result.source.value, "fallback")
                self.assertIsNone(result.rewritten_query)
                self.assertEqual(llm.calls, 1)


class IntentRoutingArbitrationTests(unittest.IsolatedAsyncioTestCase):
    """验证规则结果经仲裁后只提交查询、数量和去重上下文。"""

    async def test_repeat_without_context_is_deterministically_clarified(self) -> None:
        llm = _CountingIntentLlm({})
        recognition = await IntentRecognitionAgent(llm=llm).run("换一批")

        decision = ConversationArbitrator().decide(recognition, None)

        self.assertEqual(decision.action, ArbitrationAction.CLARIFY)
        self.assertEqual(llm.calls, 0)


class KnowledgePreparedQueryTests(unittest.IsolatedAsyncioTestCase):
    """验证主聊天查询不被重写，独立问答执行一次统一分析。"""

    async def asyncSetUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addAsyncCleanup(self._cleanup)
        self.repository = SQLiteKnowledgeRepository(
            Path(self.temporary_directory.name) / "knowledge.sqlite3"
        )

    async def _cleanup(self) -> None:
        self.temporary_directory.cleanup()

    async def test_prepared_query_is_analyzed_without_being_rewritten(self) -> None:
        self.assertIn(
            "prepared_query",
            inspect.signature(KnowledgeQaService.ask).parameters,
        )
        analyzer = _RecordingQueryAnalyzer()
        search = _RecordingSearch()
        service = KnowledgeQaService(
            repository=self.repository,
            search=search,
            query_analysis_agent=analyzer,
        )

        await service.ask(
            "为什么会失效？",
            prepared_query="Spring 事务为什么会失效？",
        )

        self.assertEqual(analyzer.calls, 1)
        self.assertEqual(search.questions, ["Spring 事务为什么会失效？"])

    async def test_independent_ask_uses_one_query_analysis(self) -> None:
        self.assertIn(
            "prepared_query",
            inspect.signature(KnowledgeQaService.ask).parameters,
        )
        analyzer = _RecordingQueryAnalyzer()
        search = _RecordingSearch()
        service = KnowledgeQaService(
            repository=self.repository,
            search=search,
            query_analysis_agent=analyzer,
        )

        await service.ask("Spring 事务为什么会失效？")

        self.assertEqual(analyzer.calls, 1)
        self.assertEqual(search.questions, ["Spring 事务为什么会失效？"])


if __name__ == "__main__":
    unittest.main()
