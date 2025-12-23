"""
Benchmarking Framework: Compare OrchestAI with baselines
"""

from typing import Dict, List, Any, Optional
from orchestai.evaluation.evaluator import Evaluator
from orchestai.evaluation.metrics import compute_metrics, compare_with_baseline


class BenchmarkSuite:
    """
    Benchmark suite for evaluating OrchestAI against baselines.
    """
    
    def __init__(
        self,
        orchestrator: Any,  # OrchestrationSystem
        baseline_systems: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize benchmark suite.
        
        Args:
            orchestrator: OrchestAI OrchestrationSystem
            baseline_systems: Dictionary of baseline systems to compare
        """
        self.orchestrator = orchestrator
        self.baseline_systems = baseline_systems or {}
    
    def run_benchmark(
        self,
        tasks: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run benchmark on set of tasks.
        
        Args:
            tasks: List of benchmark tasks
            metrics: List of metrics to compute
            
        Returns:
            Benchmark results
        """
        metrics = metrics or [
            "task_success_rate",
            "avg_cost",
            "avg_latency_ms",
            "workflow_validity",
        ]
        
        # Evaluate OrchestAI
        evaluator = Evaluator(self.orchestrator, tasks)
        orchestai_results = evaluator.evaluate(return_detailed=True)
        
        results = {
            "orchestai": {
                "metrics": orchestai_results["metrics"],
                "results": orchestai_results["results"],
            },
            "baselines": {},
        }
        
        # Evaluate baselines
        for baseline_name, baseline_system in self.baseline_systems.items():
            baseline_evaluator = Evaluator(baseline_system, tasks)
            baseline_results = baseline_evaluator.evaluate(return_detailed=True)
            results["baselines"][baseline_name] = {
                "metrics": baseline_results["metrics"],
                "results": baseline_results["results"],
            }
        
        # Compute comparisons
        if self.baseline_systems:
            comparisons = {}
            for baseline_name in self.baseline_systems.keys():
                comparison = compare_with_baseline(
                    orchestai_results["results"],
                    results["baselines"][baseline_name]["results"],
                )
                comparisons[baseline_name] = comparison
            
            results["comparisons"] = comparisons
        
        return results
    
    def generate_report(
        self,
        benchmark_results: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate human-readable benchmark report.
        
        Args:
            benchmark_results: Results from run_benchmark
            output_path: Optional path to save report
            
        Returns:
            Report string
        """
        report_lines = ["=" * 80]
        report_lines.append("ORCHESTAI BENCHMARK REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # OrchestAI results
        report_lines.append("ORCHESTAI RESULTS:")
        report_lines.append("-" * 80)
        metrics = benchmark_results["orchestai"]["metrics"]
        for key, value in metrics.items():
            report_lines.append(f"  {key}: {value:.4f}")
        report_lines.append("")
        
        # Baseline comparisons
        if "comparisons" in benchmark_results:
            report_lines.append("COMPARISONS WITH BASELINES:")
            report_lines.append("-" * 80)
            for baseline_name, comparison in benchmark_results["comparisons"].items():
                report_lines.append(f"\n{baseline_name.upper()}:")
                improvements = comparison.get("improvements", {})
                for metric, improvement in improvements.items():
                    sign = "+" if improvement > 0 else ""
                    report_lines.append(f"  {metric}: {sign}{improvement:.2f}%")
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
        
        return report

