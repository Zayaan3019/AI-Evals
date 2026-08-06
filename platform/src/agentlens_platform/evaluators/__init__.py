"""Evaluator registry: four tiers plus sandboxed custom code."""

from .base import DeterministicEvaluator, EvalInput, Evaluator, Score, Tier
from .human import HumanLabelEvaluator
from .judges import JudgeUnavailable, build_judge, ollama_available
from .registry import (
    BUILTIN_EVALUATORS,
    NonDeterministicEvaluator,
    UnknownEvaluator,
    assert_deterministic,
    build_evaluator,
    describe_registry,
    run_evaluator,
    run_suite,
)
from .sandbox import CustomEvaluator, SandboxViolation, run_sandboxed

__all__ = [
    "BUILTIN_EVALUATORS",
    "CustomEvaluator",
    "DeterministicEvaluator",
    "EvalInput",
    "Evaluator",
    "HumanLabelEvaluator",
    "JudgeUnavailable",
    "NonDeterministicEvaluator",
    "SandboxViolation",
    "Score",
    "Tier",
    "UnknownEvaluator",
    "assert_deterministic",
    "build_evaluator",
    "build_judge",
    "describe_registry",
    "ollama_available",
    "run_evaluator",
    "run_sandboxed",
    "run_suite",
]
