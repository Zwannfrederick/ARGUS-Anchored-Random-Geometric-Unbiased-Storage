"""
Hermes Adaptive Cognitive Router (Dual-Gear Reasoning Engine)
============================================================
Dynamically modulates Gemma 4's internal reasoning budget:
- LOW Risk   -> Thinking OFF (budget: 0)  -> Instantaneous sub-second (<1s) reaction.
- MEDIUM Risk -> Thinking ON (budget: -1) -> Deep cognitive architecture reasoning.
- HIGH Risk  -> Thinking ON (budget: -1) -> Safety reasoning + mandatory human approval.

Rule: Never place arbitrary token limits (max_tokens: -1 / unrestricted).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class RiskAssessment:
    level: str  # "low", "medium", "high"
    thinking_enabled: bool
    reasoning_budget: int  # 0 for immediate end, -1 for unrestricted
    requires_approval: bool
    reasons: List[str] = field(default_factory=list)


class CognitiveRouter:
    # High-risk patterns (destructive, credentials, system alteration)
    HIGH_RISK_PATTERNS = [
        r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*\s+|--recursive\s+|-[a-zA-Z]*f\s+)",
        r"\bdrop\s+table\b",
        r"\bdrop\s+database\b",
        r"\btruncate\b",
        r"\bdelete\s+from\b",
        r"\bformat\b",
        r"\bmkfs\b",
        r"\bfdisk\b",
        r"\bkill\s+-9\b",
        r"\bgit\s+push\s+.*--force\b",
        r"\bgit\s+reset\s+--hard\b",
        r"\bgit\s+clean\s+-[a-zA-Z]*f\b",
        r"\b(api[_-]?key|secret|password|credential|token)\b",
        r"\b(projeyi\s+sıfırla|tümünü\s+sil|tabloları\s+drop|diski\s+temizle)\b",
    ]

    # Medium-risk patterns (architectural changes, multi-step planning, git mutations)
    MEDIUM_RISK_PATTERNS = [
        r"\b(dag|workflow|pipeline|refactor|migration|architecture|oauth|jwt)\b",
        r"\b(git\s+commit|git\s+merge|git\s+rebase|git\s+checkout\s+-b)\b",
        r"\b(create_task_dag|plan|reorganize)\b",
        r"\b(çoklu\s+modül|mimari\s+düzenleme|bağımlılık\s+grafı)\b",
    ]

    def evaluate_risk(self, prompt: str, terminal_context: Optional[str] = None, dom_context: Optional[str] = None) -> RiskAssessment:
        """
        Evaluates risk profile of the incoming action/goal and determines cognitive gear.
        """
        combined_text = f"{prompt}\n{terminal_context or ''}\n{dom_context or ''}".lower()
        reasons: List[str] = []

        # Check for High / Destructive Risk
        for pattern in self.HIGH_RISK_PATTERNS:
            if re.search(pattern, combined_text, re.IGNORECASE):
                reasons.append(f"Detected high-risk destructive pattern: {pattern}")
                return RiskAssessment(
                    level="high",
                    thinking_enabled=True,
                    reasoning_budget=-1,  # Unrestricted deep safety reasoning
                    requires_approval=True,
                    reasons=reasons,
                )

        # Check for Medium Risk (Architectural / Multi-step / Code mutation)
        for pattern in self.MEDIUM_RISK_PATTERNS:
            if re.search(pattern, combined_text, re.IGNORECASE):
                reasons.append(f"Detected medium-risk planning pattern: {pattern}")
                return RiskAssessment(
                    level="medium",
                    thinking_enabled=True,
                    reasoning_budget=-1,  # Unrestricted architectural reasoning
                    requires_approval=False,
                    reasons=reasons,
                )

        # Default: Low Risk (Immediate execution, browsing, read-only inspection)
        reasons.append("Routine action / read-only / low-risk operation")
        return RiskAssessment(
            level="low",
            thinking_enabled=False,
            reasoning_budget=0,  # Thinking bypassed: <1s action latency!
            requires_approval=False,
            reasons=reasons,
        )
