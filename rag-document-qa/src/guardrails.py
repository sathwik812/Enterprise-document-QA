"""
Hallucination detection guardrail using RAGAS faithfulness metric.

Falls back to a keyword-overlap heuristic when RAGAS is unavailable.
"""

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _faithfulness_threshold() -> float:
    try:
        return float(os.getenv("FAITHFULNESS_THRESHOLD", "0.7"))
    except ValueError:
        return 0.7


# ---------------------------------------------------------------------------
# Keyword-overlap fallback
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Lower-case word tokenisation, stripping punctuation."""
    tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
    stopwords = {"the", "and", "are", "was", "were", "for", "not", "but", "with", "from"}
    return {t for t in tokens if t not in stopwords}


def _keyword_overlap_score(answer: str, contexts: list[str]) -> float:
    """
    Simple keyword overlap heuristic.

    Score = |answer_tokens ∩ context_tokens| / max(|answer_tokens|, 1)
    """
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 0.0

    context_text = " ".join(contexts)
    context_tokens = _tokenize(context_text)

    overlap = answer_tokens & context_tokens
    return len(overlap) / len(answer_tokens)


# ---------------------------------------------------------------------------
# RAGAS-based scoring
# ---------------------------------------------------------------------------

def _ragas_faithfulness_score(
    question: str, answer: str, contexts: list[str]
) -> float | None:
    """
    Compute RAGAS faithfulness score.

    Returns a float in [0, 1] or None if RAGAS is unavailable / errors.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=[faithfulness])
        score = result["faithfulness"]
        # result may be a list or a scalar depending on RAGAS version
        if isinstance(score, (list, tuple)):
            score = score[0]
        return float(score)
    except ImportError:
        logger.debug("RAGAS not available; using keyword-overlap fallback.")
        return None
    except Exception as exc:
        logger.warning("RAGAS evaluation failed (%s); using keyword-overlap fallback.", exc)
        return None


# ---------------------------------------------------------------------------
# HallucinationGuard
# ---------------------------------------------------------------------------

class HallucinationGuard:
    """
    Checks whether an LLM answer is faithful to the retrieved contexts.

    Usage
    -----
    guard = HallucinationGuard()
    result = guard.check(question, answer, contexts)
    # result = {"passed": True, "score": 0.85, "reason": "..."}
    """

    def __init__(self) -> None:
        self.threshold = _faithfulness_threshold()
        logger.info("HallucinationGuard initialised with threshold=%.2f", self.threshold)

    def check(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> dict[str, Any]:
        """
        Evaluate whether *answer* is grounded in *contexts*.

        Parameters
        ----------
        question : str
            The original user question.
        answer : str
            The LLM-generated answer to evaluate.
        contexts : list[str]
            The retrieved context chunks used to generate the answer.

        Returns
        -------
        dict with keys:
            passed  – bool, True if score >= threshold
            score   – float in [0, 1]
            reason  – human-readable explanation
        """
        if not answer or not answer.strip():
            return {
                "passed": False,
                "score": 0.0,
                "reason": "Empty answer cannot be verified.",
            }

        if not contexts:
            return {
                "passed": False,
                "score": 0.0,
                "reason": "No context provided; cannot verify faithfulness.",
            }

        # Try RAGAS first
        score = _ragas_faithfulness_score(question, answer, contexts)
        method = "RAGAS faithfulness"

        if score is None:
            # Fall back to keyword overlap
            score = _keyword_overlap_score(answer, contexts)
            method = "keyword-overlap heuristic"

        passed = score >= self.threshold

        if passed:
            reason = (
                f"Answer is sufficiently grounded in context "
                f"({method} score: {score:.3f} ≥ threshold {self.threshold:.2f})."
            )
        else:
            reason = (
                f"Answer may contain hallucinations "
                f"({method} score: {score:.3f} < threshold {self.threshold:.2f}). "
                "Response blocked."
            )

        logger.info(
            "Guardrail check — method: %s, score: %.3f, passed: %s",
            method,
            score,
            passed,
        )

        return {"passed": passed, "score": score, "reason": reason}
