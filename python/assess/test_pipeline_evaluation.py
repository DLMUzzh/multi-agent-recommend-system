"""SQLite Chunk 统一推荐评估的定向验证。"""

from __future__ import annotations

import importlib
import json
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch


class PipelineEvaluationTests(unittest.IsolatedAsyncioTestCase):
    """验证评估只使用真实 SQLite 文档和统一 Chunk 召回组件。"""

    def _module(self) -> Any:
        try:
            return importlib.import_module("assess.pipeline_evaluation")
        except ModuleNotFoundError:
            self.fail("缺少 assess.pipeline_evaluation")

    def test_fixed_cases_use_query_and_document_ids(self) -> None:
        module = self._module()
        path = Path(__file__).resolve().parents[2] / "data" / "evaluation_cases.json"

        self.assertTrue(path.exists(), "缺少 data/evaluation_cases.json")
        cases = module.load_pipeline_cases(path)

        self.assertEqual(len(cases), 7)
        self.assertEqual(cases[0].query_id, "java_virtual_threads")
        self.assertEqual(cases[0].query, "Java 虚拟线程适合什么场景")
        self.assertEqual(cases[0].relevance, {"java-post-002": 3.0})
        self.assertEqual(cases[-1].query_id, "feishu_knowledge_qa")
        self.assertEqual(
            cases[-1].relevance,
            {"IP5Ad3RFNosCXoxuB1bcm5lVnvf": 3.0},
        )
        self.assertFalse(hasattr(cases[0], "user_id"))
        self.assertFalse(hasattr(cases[0], "context"))
        self.assertFalse(hasattr(cases[0], "profile_blocked_topics"))

    def test_assess_package_exports_pipeline_entrypoints(self) -> None:
        assess = importlib.import_module("assess")

        self.assertTrue(hasattr(assess, "evaluate_pipeline"))
        self.assertTrue(hasattr(assess, "load_pipeline_cases"))
        self.assertTrue(hasattr(assess, "PipelineVariantReport"))

    async def test_evaluation_runs_sqlite_recall_and_document_aggregation(self) -> None:
        module = self._module()

        report = await module.evaluate_pipeline(k=5)

        self.assertEqual(report.evaluated_queries, 7)
        self.assertEqual(report.recall.latency.sample_count, 7)
        self.assertEqual(report.rerank.latency.sample_count, 7)
        self.assertEqual(report.recall.llm_call_count, 0)
        self.assertEqual(report.rerank.llm_call_count, 0)
        self.assertEqual(report.rerank.degraded_query_count, 7)
        self.assertGreaterEqual(report.recall.hit_at_k.value, 0.85)
        self.assertGreaterEqual(report.recall.mrr_at_k.value, 0.70)
        self.assertEqual(report.recall.violation_rate.violation_rate, 0.0)
        self.assertEqual(report.rerank.violation_rate.violation_rate, 0.0)
        self.assertTrue(
            all(mode == "bm25" for mode in report.retrieval_modes.values())
        )
        self.assertTrue(report.used_real_pipeline_components)
        self.assertFalse(report.used_external_services)
        self.assertFalse(hasattr(report, "fact_profile_rerank"))
        self.assertFalse(hasattr(report, "semantic_profile_rerank"))

    async def test_fake_llm_only_receives_chunk_document_evidence(self) -> None:
        module = self._module()
        rerank_llm = _ReverseDocumentRerankLlm()

        report = await module.evaluate_pipeline(k=5, rerank_llm=rerank_llm)

        self.assertEqual(rerank_llm.calls, 7)
        self.assertEqual(report.rerank.llm_call_count, 7)
        self.assertEqual(report.rerank.degraded_query_count, 0)
        self.assertFalse(report.used_external_services)
        self.assertTrue(rerank_llm.candidate_payloads)
        for candidates in rerank_llm.candidate_payloads:
            self.assertTrue(candidates)
            for candidate in candidates:
                self.assertEqual(
                    set(candidate),
                    {"document_id", "title", "excerpt", "recall_score"},
                )

    async def test_evaluation_ignores_legacy_recall_environment(self) -> None:
        module = self._module()
        hostile_environment = {
            "ARTICLE_REC_RECALL_BM25_TITLE_WEIGHT": "0",
            "ARTICLE_REC_RECALL_BM25_TOPIC_WEIGHT": "0",
            "ARTICLE_REC_RECALL_BM25_DESCRIPTION_WEIGHT": "0",
            "ARTICLE_REC_RECALL_MODE": "hybrid",
        }

        with patch.dict("os.environ", hostile_environment):
            report = await module.evaluate_pipeline(k=5)

        self.assertGreaterEqual(report.recall.hit_at_k.value, 0.85)
        self.assertTrue(
            all(mode == "bm25" for mode in report.retrieval_modes.values())
        )

    def test_invalid_or_duplicate_cases_are_rejected(self) -> None:
        module = self._module()
        with self.subTest("旧画像字段被拒绝"):
            path = self._write_case_file(
                [
                    {
                        "query_id": "legacy",
                        "query": "Java",
                        "size": 5,
                        "relevance": {"java-post-002": 3.0},
                        "profile_blocked_topics": ["娱乐"],
                    }
                ]
            )
            self.addCleanup(path.unlink)
            with self.assertRaisesRegex(ValueError, "第 1 条无效"):
                module.load_pipeline_cases(path)

        with self.subTest("query_id 不能重复"):
            row = {
                "query_id": "duplicate",
                "query": "Java",
                "size": 5,
                "relevance": {"java-post-002": 3.0},
            }
            path = self._write_case_file([row, row])
            self.addCleanup(path.unlink)
            with self.assertRaisesRegex(ValueError, "query_id 不能重复"):
                module.load_pipeline_cases(path)

    @staticmethod
    def _write_case_file(payload: object) -> Path:
        import tempfile

        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            delete=False,
        )
        with handle:
            json.dump(payload, handle, ensure_ascii=False)
        return Path(handle.name)


class _ReverseDocumentRerankLlm:
    """读取最小证据投影并返回完整、反向的受控文档评分。"""

    def __init__(self) -> None:
        self.calls = 0
        self.candidate_payloads: list[list[dict[str, Any]]] = []

    async def ainvoke(self, messages: list[Any]) -> dict[str, Any]:
        self.calls += 1
        content = str(messages[-1].content)
        envelope = json.loads(content)
        payload = envelope["input"]
        candidates = payload["candidates"]
        self.candidate_payloads.append(candidates)
        return {
            "items": [
                {
                    "document_id": item["document_id"],
                    "llm_score": round(
                        1.0 - index / max(len(candidates), 1),
                        6,
                    ),
                    "reason": "命中摘录与查询相关",
                }
                for index, item in enumerate(reversed(candidates))
            ]
        }


if __name__ == "__main__":
    unittest.main()
