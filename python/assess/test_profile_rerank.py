"""验证 SQLite 用户画像与文档软重排。"""

from __future__ import annotations

import asyncio
import json
import unittest
from datetime import datetime, timezone
from typing import Any

from app.agents.document_rerank_agent import DocumentRerankAgent
from app.agents.user_profile_agent import UserProfileAgent
from app.config import Settings
from app.infrastructure.database.json.feature_store import FeatureStore
from app.models.schemas import DocumentCandidate


class _FixedLlm:
    def __init__(self, output: dict[str, Any]) -> None:
        self.output = output
        self.messages: list[Any] = []

    async def ainvoke(self, messages: list[Any]) -> dict[str, Any]:
        self.messages = list(messages)
        return self.output


class SQLiteUserProfileAgentTests(unittest.IsolatedAsyncioTestCase):
    """SQLite 事实由画像 Agent 归一化为长度偏好和最终置信度。"""

    async def test_agent_owns_profile_confidence_and_reading_length(self) -> None:
        def clock() -> datetime:
            return datetime(2026, 8, 8, tzinfo=timezone.utc)

        store = FeatureStore(clock=clock)

        features = await store.get_user_features(
            "10001",
            as_of="2026-08-08T00:00:00+00:00",
        )
        result = await UserProfileAgent(
            feature_store=store,
            enable_llm=False,
            clock=clock,
            settings=Settings(_env_file=None),
        ).run(user_id="10001")

        self.assertNotIn("profile_confidence", features)
        self.assertIn("confidence_inputs", features)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.profile)
        assert result.profile is not None
        self.assertGreater(result.profile.profile_confidence, 0.5)
        self.assertTrue(
            result.profile.behavior_profile.reading_length_preferences
        )


class SQLiteDocumentProfileRerankTests(unittest.IsolatedAsyncioTestCase):
    """文档重排以动态 8:2 软融合画像且不向 LLM 暴露画像。"""

    async def test_profile_soft_rerank_uses_approved_dimensions(self) -> None:
        def clock() -> datetime:
            return datetime(2026, 8, 8, tzinfo=timezone.utc)

        profile_result = await UserProfileAgent(
            feature_store=FeatureStore(clock=clock),
            enable_llm=False,
            clock=clock,
            settings=Settings(_env_file=None),
        ).run(user_id="10001")
        assert profile_result.profile is not None
        llm = _FixedLlm(
            {
                "items": [
                    {
                        "document_id": document_id,
                        "llm_score": 0.8,
                        "reason": "与查询相关。",
                    }
                    for document_id in ("doc-fit", "doc-other")
                ]
            }
        )
        candidates = [
            DocumentCandidate(
                document_id="doc-fit",
                title="Spring Boot 实践",
                topics=["Spring Boot"],
                content_type="tutorial",
                difficulty="intermediate",
                author_id="author-fit",
                total_token_count=1200,
                excerpt="Spring Boot 部署与配置实践。",
                matched_chunk_ids=["chunk-fit"],
                recall_score=0.8,
            ),
            DocumentCandidate(
                document_id="doc-other",
                title="高级分析",
                topics=["其他主题"],
                content_type="analysis",
                difficulty="advanced",
                author_id="author-other",
                total_token_count=4000,
                excerpt="其他主题的深入分析。",
                matched_chunk_ids=["chunk-other"],
                recall_score=0.8,
            ),
        ]

        result = await DocumentRerankAgent(llm=llm).run(
            query="推荐 Spring Boot 实践",
            candidates=candidates,
            user_profile=profile_result.profile,
            current_topics=["Spring Boot"],
        )

        self.assertEqual(
            [item.document_id for item in result.ranked_documents],
            ["doc-fit", "doc-other"],
        )
        self.assertGreater(
            result.ranked_documents[0].profile_score,
            result.ranked_documents[1].profile_score,
        )
        self.assertAlmostEqual(
            result.data["blend_weights"]["profile_weight"],
            0.20 * profile_result.profile.profile_confidence,
        )
        envelope = json.loads(str(llm.messages[-1].content))
        self.assertEqual(set(envelope), {"contract", "input"})
        self.assertEqual(envelope["contract"]["name"], "document_rerank")
        self.assertEqual(envelope["contract"]["version"], 2)
        self.assertIsInstance(envelope["contract"]["output_schema"], dict)
        schema_text = json.dumps(
            envelope["contract"]["output_schema"],
            ensure_ascii=False,
            sort_keys=True,
        )
        self.assertIn('"document_id"', schema_text)
        self.assertIn('"llm_score"', schema_text)
        self.assertIn('"reason"', schema_text)
        payload = envelope["input"]
        self.assertEqual(
            set(payload["candidates"][0]),
            {"document_id", "title", "excerpt", "recall_score"},
        )

    async def test_llm_prompt_defines_evidence_score_anchors(self) -> None:
        llm = _FixedLlm(
            {
                "items": [
                    {
                        "document_id": "doc-a",
                        "llm_score": 0.5,
                        "reason": "摘录只支持查询的一部分。",
                    }
                ]
            }
        )
        candidate = DocumentCandidate(
            document_id="doc-a",
            title="Spring Boot 部署",
            topics=["Spring Boot"],
            content_type="tutorial",
            difficulty="intermediate",
            author_id="author-a",
            total_token_count=1200,
            excerpt="说明多实例部署时如何避免定时任务重复执行。",
            matched_chunk_ids=["chunk-a"],
            recall_score=0.8,
        )

        result = await DocumentRerankAgent(llm=llm).run(
            query="Spring Boot 如何完整部署？",
            candidates=[candidate],
        )

        self.assertTrue(result.success)
        prompt = str(llm.messages[0].content)
        self.assertIn("1.0", prompt)
        self.assertIn("0.5", prompt)
        self.assertIn("0.0", prompt)
        self.assertIn("仅标题词面重合", prompt)
        self.assertIn("召回分只作参考", prompt)

    async def test_incomplete_small_rerank_upgrades_to_large_once(self) -> None:
        from app.infrastructure.llm.client import llm_upgrade_scope

        candidates = [
            DocumentCandidate(
                document_id=document_id,
                title=f"文档 {document_id}",
                topics=["Spring"],
                content_type="tutorial",
                difficulty="intermediate",
                author_id=f"author-{document_id}",
                total_token_count=800,
                excerpt=f"{document_id} 的可信摘录",
                matched_chunk_ids=[f"chunk-{document_id}"],
                recall_score=score,
            )
            for document_id, score in (("doc-a", 0.8), ("doc-b", 0.7))
        ]
        small = _FixedLlm(
            {
                "items": [
                    {
                        "document_id": "doc-a",
                        "llm_score": 0.8,
                        "reason": "输出不完整。",
                    }
                ]
            }
        )
        large = _FixedLlm(
            {
                "items": [
                    {
                        "document_id": "doc-a",
                        "llm_score": 0.4,
                        "reason": "部分相关。",
                    },
                    {
                        "document_id": "doc-b",
                        "llm_score": 0.9,
                        "reason": "直接相关。",
                    },
                ]
            }
        )
        agent = DocumentRerankAgent(llm=small, large_llm=large)

        deadline = asyncio.get_running_loop().time() + 60.0
        with llm_upgrade_scope(deadline=deadline):
            result = await agent.run(
                query="需要 doc-b 的内容",
                candidates=candidates,
            )

        self.assertEqual(len(small.messages), 2)
        self.assertEqual(len(large.messages), 2)
        self.assertEqual(result.data["llm_status"], "upgraded")
        self.assertEqual(result.data["llm_call_count"], 2)
        self.assertEqual(result.ranked_documents[0].document_id, "doc-b")

    async def test_incomplete_large_rerank_still_reports_two_llm_calls(self) -> None:
        from app.infrastructure.llm.client import llm_upgrade_scope

        candidates = [
            DocumentCandidate(
                document_id=document_id,
                title=f"文档 {document_id}",
                topics=["Spring"],
                content_type="tutorial",
                difficulty="intermediate",
                author_id=f"author-{document_id}",
                total_token_count=800,
                excerpt=f"{document_id} 的可信摘录",
                matched_chunk_ids=[f"chunk-{document_id}"],
                recall_score=score,
            )
            for document_id, score in (("doc-a", 0.8), ("doc-b", 0.7))
        ]
        small = _FixedLlm(
            {
                "items": [
                    {
                        "document_id": "doc-a",
                        "llm_score": 0.8,
                        "reason": "小模型输出不完整。",
                    }
                ]
            }
        )
        large = _FixedLlm(
            {
                "items": [
                    {
                        "document_id": "doc-b",
                        "llm_score": 0.9,
                        "reason": "大模型输出仍不完整。",
                    }
                ]
            }
        )
        agent = DocumentRerankAgent(llm=small, large_llm=large)

        deadline = asyncio.get_running_loop().time() + 60.0
        with llm_upgrade_scope(deadline=deadline):
            result = await agent.run(
                query="需要完整比较两个文档",
                candidates=candidates,
            )

        self.assertEqual(len(small.messages), 2)
        self.assertEqual(len(large.messages), 2)
        self.assertEqual(result.data["llm_status"], "discarded_incomplete_batch")
        self.assertEqual(result.data["llm_call_count"], 2)
        self.assertFalse(result.data["llm_applied"])


if __name__ == "__main__":
    unittest.main()
