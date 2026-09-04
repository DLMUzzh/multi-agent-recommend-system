"""推荐系统纯评估方法的定向自验证。"""

from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError

import assess
from assess.diagnostics import diagnose
from assess.metrics import (
    degradation_success_rate,
    hit_rate_at_k,
    latency_summary,
    metric_lift,
    mrr_at_k,
    ndcg_at_k,
    parallel_summary,
    precision_at_k,
    recall_at_k,
    violation_rate,
)
from assess.models import (
    DiagnosticFinding,
    EvaluationCase,
    LatencySummary,
    LiftSummary,
    MetricResult,
    ParallelSummary,
    SuccessSummary,
    ViolationSummary,
)


class EvaluationModelTests(unittest.TestCase):
    """验证评估输入和结果模型的不变量。"""

    def test_evaluation_case_copies_inputs_and_excludes_forbidden_relevant_ids(
        self,
    ) -> None:
        relevance = {"article-1": 3, "article-2": 2, "article-3": 1}
        forbidden = {"article-2"}

        case = EvaluationCase(
            query_id="query-1",
            relevance=relevance,
            forbidden_ids=forbidden,
        )
        relevance["article-1"] = 0
        forbidden.add("article-1")

        self.assertEqual(
            dict(case.relevance),
            {"article-1": 3.0, "article-2": 2.0, "article-3": 1.0},
        )
        self.assertEqual(case.forbidden_ids, frozenset({"article-2"}))
        self.assertEqual(case.relevant_ids(), frozenset({"article-1"}))
        self.assertEqual(
            case.relevant_ids(relevance_threshold=1.0),
            frozenset({"article-1", "article-3"}),
        )
        with self.assertRaises(TypeError):
            case.relevance["article-4"] = 3.0  # type: ignore[index]

    def test_evaluation_case_rejects_invalid_ids_and_relevance(self) -> None:
        invalid_cases = (
            {"query_id": "", "relevance": {"a": 3}},
            {"query_id": "q", "relevance": {"": 3}},
            {"query_id": "q", "relevance": {"a": True}},
            {"query_id": "q", "relevance": {"a": math.nan}},
            {"query_id": "q", "relevance": {"a": math.inf}},
            {"query_id": "q", "relevance": {"a": -0.1}},
            {"query_id": "q", "relevance": {"a": 3.1}},
            {
                "query_id": "q",
                "relevance": {"a": 3},
                "forbidden_ids": {""},
            },
        )

        for parameters in invalid_cases:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    EvaluationCase(**parameters)

    def test_result_models_are_immutable_and_validate_counts(self) -> None:
        result = MetricResult(value=0.75, evaluated_queries=3, skipped_queries=1)

        with self.assertRaises(FrozenInstanceError):
            result.value = 0.5  # type: ignore[misc]
        with self.assertRaises(ValueError):
            MetricResult(value=math.nan, evaluated_queries=1)
        with self.assertRaises(ValueError):
            MetricResult(value=0.5, evaluated_queries=-1)

    def test_all_result_models_accept_valid_values(self) -> None:
        latency = LatencySummary(
            sample_count=3,
            mean_ms=20.0,
            p50_ms=20.0,
            p95_ms=29.0,
            p99_ms=29.8,
            max_ms=30.0,
        )
        parallel = ParallelSummary(
            bm25_ms=300.0,
            vector_ms=500.0,
            hybrid_ms=550.0,
            sequential_ms=800.0,
            saving_ms=250.0,
            saving_rate=0.3125,
            speedup=800.0 / 550.0,
            overlap_efficiency=250.0 / 300.0,
        )
        lift = LiftSummary(
            baseline=0.8,
            candidate=0.9,
            absolute_lift=0.1,
            relative_lift=0.125,
        )
        violation = ViolationSummary(
            violation_rate=0.2,
            violation_count=1,
            result_count=5,
        )
        success = SuccessSummary(success_rate=0.75, success_count=3, total_count=4)
        finding = DiagnosticFinding(
            code="low_hit",
            severity="high",
            message="优先检查召回覆盖。",
        )

        self.assertEqual(latency.sample_count, 3)
        self.assertGreater(parallel.overlap_efficiency, 0.8)
        self.assertAlmostEqual(lift.absolute_lift, 0.1)
        self.assertEqual(violation.violation_count, 1)
        self.assertEqual(success.total_count, 4)
        self.assertEqual(finding.code, "low_hit")


class RankingMetricTests(unittest.TestCase):
    """验证文章级或 Chunk 级排名指标。"""

    def setUp(self) -> None:
        self.cases = (
            EvaluationCase(
                query_id="q1",
                relevance={"a": 3, "b": 2, "c": 1},
            ),
            EvaluationCase(
                query_id="q2",
                relevance={"d": 3, "e": 2},
            ),
            EvaluationCase(
                query_id="q3",
                relevance={"f": 1},
            ),
        )
        self.rankings = {
            "q1": ["x", "b", "a"],
            "q2": ["x", "y", "d"],
            "q3": ["f"],
        }

    def test_hit_recall_precision_and_mrr_use_expected_denominators(self) -> None:
        hit = hit_rate_at_k(self.cases, self.rankings, k=3)
        recall = recall_at_k(self.cases, self.rankings, k=3)
        precision = precision_at_k(self.cases, self.rankings, k=3)
        mrr = mrr_at_k(self.cases, self.rankings, k=3)

        self.assertEqual(hit, MetricResult(1.0, evaluated_queries=2, skipped_queries=1))
        self.assertEqual(
            recall,
            MetricResult(0.75, evaluated_queries=2, skipped_queries=1),
        )
        self.assertEqual(
            precision,
            MetricResult(1.0 / 3.0, evaluated_queries=3, skipped_queries=0),
        )
        self.assertEqual(mrr.evaluated_queries, 2)
        self.assertEqual(mrr.skipped_queries, 1)
        self.assertAlmostEqual(mrr.value, (1.0 / 2.0 + 1.0 / 3.0) / 2.0)

    def test_precision_uses_k_when_ranking_contains_fewer_results(self) -> None:
        case = EvaluationCase(query_id="q", relevance={"a": 3})

        result = precision_at_k((case,), {"q": ["a"]}, k=5)

        self.assertEqual(result, MetricResult(0.2, 1, 0))

    def test_forbidden_relevant_item_is_not_a_valid_hit_or_ndcg_gain(self) -> None:
        case = EvaluationCase(
            query_id="q",
            relevance={"a": 3, "b": 3},
            forbidden_ids={"a"},
        )
        rankings = {"q": ["a", "b"]}

        self.assertEqual(hit_rate_at_k((case,), rankings, k=1).value, 0.0)
        self.assertEqual(hit_rate_at_k((case,), rankings, k=2).value, 1.0)
        self.assertAlmostEqual(
            ndcg_at_k((case,), rankings, k=2).value,
            1.0 / math.log2(3.0),
        )

    def test_ndcg_uses_graded_relevance_and_skips_zero_gain_queries(self) -> None:
        cases = (
            EvaluationCase(query_id="graded", relevance={"a": 3, "b": 2}),
            EvaluationCase(query_id="zero", relevance={"z": 0}),
        )
        rankings = {"graded": ["b", "a"], "zero": []}
        actual_dcg = 3.0 + 7.0 / math.log2(3.0)
        ideal_dcg = 7.0 + 3.0 / math.log2(3.0)

        result = ndcg_at_k(cases, rankings, k=2)

        self.assertAlmostEqual(result.value, actual_dcg / ideal_dcg)
        self.assertEqual(result.evaluated_queries, 1)
        self.assertEqual(result.skipped_queries, 1)

    def test_missing_ranking_is_treated_as_empty_without_modifying_inputs(self) -> None:
        rankings = {"q1": ["a"]}
        snapshot = {key: list(values) for key, values in rankings.items()}

        result = hit_rate_at_k(self.cases, rankings, k=3)

        self.assertAlmostEqual(result.value, 0.5)
        self.assertEqual(rankings, snapshot)

    def test_ranking_metrics_reject_invalid_k_duplicate_ids_and_unknown_queries(
        self,
    ) -> None:
        invalid_calls = (
            lambda: hit_rate_at_k(self.cases, self.rankings, k=0),
            lambda: recall_at_k(
                self.cases,
                {**self.rankings, "q1": ["a", "a"]},
                k=3,
            ),
            lambda: precision_at_k(
                self.cases,
                {**self.rankings, "unknown": ["a"]},
                k=3,
            ),
            lambda: mrr_at_k(
                (*self.cases, self.cases[0]),
                self.rankings,
                k=3,
            ),
            lambda: ndcg_at_k(
                self.cases,
                {**self.rankings, "q2": [""]},
                k=3,
            ),
        )

        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()


class PerformanceAndBoundaryMetricTests(unittest.TestCase):
    """验证延迟、并行、提升、违规和降级指标。"""

    def test_metric_lift_reports_absolute_and_optional_relative_lift(self) -> None:
        regular = metric_lift(baseline=0.8, candidate=0.9)
        zero_baseline = metric_lift(baseline=0.0, candidate=0.1)

        self.assertAlmostEqual(regular.absolute_lift, 0.1)
        self.assertAlmostEqual(regular.relative_lift or 0.0, 0.125)
        self.assertEqual(zero_baseline.absolute_lift, 0.1)
        self.assertIsNone(zero_baseline.relative_lift)

    def test_latency_summary_uses_linear_interpolation(self) -> None:
        result = latency_summary([10.0, 20.0, 30.0, 40.0])

        self.assertEqual(result.sample_count, 4)
        self.assertEqual(result.mean_ms, 25.0)
        self.assertEqual(result.p50_ms, 25.0)
        self.assertAlmostEqual(result.p95_ms, 38.5)
        self.assertAlmostEqual(result.p99_ms, 39.7)
        self.assertEqual(result.max_ms, 40.0)

    def test_latency_summary_handles_single_sample_and_rejects_invalid_samples(
        self,
    ) -> None:
        result = latency_summary([12])

        self.assertEqual(
            result,
            LatencySummary(1, 12.0, 12.0, 12.0, 12.0, 12.0),
        )
        for samples in ([], [-1], [math.nan], [math.inf], [True]):
            with self.subTest(samples=samples):
                with self.assertRaises(ValueError):
                    latency_summary(samples)

    def test_parallel_summary_quantifies_saving_and_preserves_negative_gain(
        self,
    ) -> None:
        effective = parallel_summary(
            bm25_ms=300,
            vector_ms=500,
            hybrid_ms=550,
        )
        slower = parallel_summary(
            bm25_ms=300,
            vector_ms=500,
            hybrid_ms=900,
        )

        self.assertEqual(effective.sequential_ms, 800.0)
        self.assertEqual(effective.saving_ms, 250.0)
        self.assertEqual(effective.saving_rate, 0.3125)
        self.assertAlmostEqual(effective.speedup, 800.0 / 550.0)
        self.assertAlmostEqual(effective.overlap_efficiency, 250.0 / 300.0)
        self.assertEqual(slower.saving_ms, -100.0)
        self.assertLess(slower.overlap_efficiency, 0.0)

    def test_parallel_summary_rejects_non_positive_or_non_finite_times(self) -> None:
        invalid_values = (0, -1, math.nan, math.inf, True)
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parallel_summary(
                        bm25_ms=value,
                        vector_ms=500,
                        hybrid_ms=550,
                    )

    def test_violation_rate_counts_forbidden_results_and_empty_output(self) -> None:
        cases = (
            EvaluationCase(
                query_id="q1",
                relevance={"a": 3},
                forbidden_ids={"x"},
            ),
            EvaluationCase(
                query_id="q2",
                relevance={"b": 3},
                forbidden_ids={"y"},
            ),
        )
        rankings = {"q1": ["x", "a"], "q2": []}

        full = violation_rate(cases, rankings)
        top_one = violation_rate(cases, rankings, k=1)
        empty = violation_rate(cases, {})

        self.assertEqual(full, ViolationSummary(0.5, 1, 2))
        self.assertEqual(top_one, ViolationSummary(1.0, 1, 1))
        self.assertEqual(empty, ViolationSummary(0.0, 0, 0))

    def test_degradation_success_rate_requires_non_empty_boolean_outcomes(self) -> None:
        result = degradation_success_rate([True, False, True, True])

        self.assertEqual(result, SuccessSummary(0.75, 3, 4))
        for outcomes in ([], [1], [True, "yes"]):
            with self.subTest(outcomes=outcomes):
                with self.assertRaises(ValueError):
                    degradation_success_rate(outcomes)

    def test_performance_metrics_reject_non_finite_lift_and_invalid_violation_k(
        self,
    ) -> None:
        case = EvaluationCase(query_id="q", relevance={"a": 3})

        with self.assertRaises(ValueError):
            metric_lift(baseline=math.nan, candidate=0.5)
        with self.assertRaises(ValueError):
            violation_rate((case,), {"q": ["a"]}, k=0)


class DiagnosticTests(unittest.TestCase):
    """验证指标组合到可能原因的确定性映射。"""

    @staticmethod
    def _codes(**overrides: float) -> list[str]:
        parameters = {
            "hit_at_k": 0.99,
            "rrf_mrr": 0.90,
            "final_mrr": 0.90,
            "final_ndcg": 0.90,
            "violation_rate": 0.0,
            "degradation_success_rate": 1.0,
        }
        parameters.update(overrides)
        return [finding.code for finding in diagnose(**parameters)]

    def test_low_hit_points_to_recall_coverage(self) -> None:
        self.assertEqual(
            self._codes(
                hit_at_k=0.80,
                rrf_mrr=0.20,
                final_mrr=0.20,
                final_ndcg=0.20,
            ),
            ["low_hit"],
        )

    def test_high_hit_and_low_rrf_mrr_points_to_retrieval_ranking(self) -> None:
        self.assertEqual(
            self._codes(rrf_mrr=0.50, final_mrr=0.50),
            ["retrieval_ranking"],
        )

    def test_rerank_regression_and_improvement_are_distinguished(self) -> None:
        regression = self._codes(rrf_mrr=0.90, final_mrr=0.70)
        improvement = [
            finding.code
            for finding in diagnose(
                hit_at_k=0.99,
                rrf_mrr=0.60,
                final_mrr=0.85,
                final_ndcg=0.90,
                violation_rate=0.0,
                degradation_success_rate=1.0,
                mrr_threshold=0.50,
            )
        ]

        self.assertEqual(regression, ["rerank_regression"])
        self.assertEqual(improvement, ["rerank_improvement"])

    def test_high_hit_and_mrr_but_low_ndcg_points_to_later_rank_quality(self) -> None:
        self.assertEqual(
            self._codes(final_ndcg=0.60),
            ["low_ndcg"],
        )

    def test_violation_and_fallback_failures_have_high_priority(self) -> None:
        findings = diagnose(
            hit_at_k=0.99,
            rrf_mrr=0.90,
            final_mrr=0.90,
            final_ndcg=0.90,
            violation_rate=0.10,
            degradation_success_rate=0.80,
        )

        self.assertEqual(
            [finding.code for finding in findings],
            ["hard_constraint_violation", "fallback_failure"],
        )
        self.assertTrue(all(finding.severity == "high" for finding in findings))

    def test_healthy_metrics_return_no_findings(self) -> None:
        self.assertEqual(self._codes(), [])

    def test_diagnose_rejects_invalid_rates_and_thresholds(self) -> None:
        invalid_calls = (
            lambda: diagnose(
                hit_at_k=1.1,
                rrf_mrr=0.9,
                final_mrr=0.9,
                final_ndcg=0.9,
                violation_rate=0.0,
                degradation_success_rate=1.0,
            ),
            lambda: diagnose(
                hit_at_k=0.9,
                rrf_mrr=math.nan,
                final_mrr=0.9,
                final_ndcg=0.9,
                violation_rate=0.0,
                degradation_success_rate=1.0,
            ),
            lambda: diagnose(
                hit_at_k=0.9,
                rrf_mrr=0.9,
                final_mrr=0.9,
                final_ndcg=0.9,
                violation_rate=0.0,
                degradation_success_rate=1.0,
                rerank_delta_threshold=-0.1,
            ),
        )

        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()


class PackageExportTests(unittest.TestCase):
    """验证评估包提供稳定且集中的导入入口。"""

    def test_package_exports_models_metrics_and_diagnostics(self) -> None:
        expected_names = {
            "EvaluationCase",
            "MetricResult",
            "hit_rate_at_k",
            "recall_at_k",
            "precision_at_k",
            "mrr_at_k",
            "ndcg_at_k",
            "metric_lift",
            "latency_summary",
            "parallel_summary",
            "violation_rate",
            "degradation_success_rate",
            "diagnose",
        }

        self.assertTrue(expected_names.issubset(set(assess.__all__)))
        self.assertTrue(all(hasattr(assess, name) for name in expected_names))


if __name__ == "__main__":
    unittest.main()
